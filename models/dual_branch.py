
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import FEATURE_DIM, MEMORY_DIM
from models.mcre import GN


class ProposalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM,
                3,
                padding=1,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM,
                3,
                padding=1,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)


class CenterHeatmapHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM,
                3,
                padding=1,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM // 2,
                3,
                padding=1,
                bias=False
            ),
            GN(FEATURE_DIM // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(FEATURE_DIM // 2, 1, 1)
        )

    def forward(self, x):
        return self.head(x)


class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM // 2,
                3,
                padding=1,
                bias=False
            ),
            GN(FEATURE_DIM // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(FEATURE_DIM // 2, 1, 1)
        )

    def forward(self, x):
        return self.head(x)


class BoundaryFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM,
                3,
                padding=1,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True),
            nn.Conv2d(FEATURE_DIM, 96, 1)
        )

    def forward(self, x):
        return self.net(x)


class AppearanceEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM,
                1,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.embed(x)


class GeometryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(96, 64, 1, bias=False),
            GN(64),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.embed(x)


class ProposalTokenGenerator(nn.Module):
    def __init__(self, token_dim=224, topk=100):
        super().__init__()
        self.topk = topk
        self.token_dim = token_dim
        self.token_projection = nn.Linear(256, MEMORY_DIM)
        self.score_head = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.LayerNorm(MEMORY_DIM // 2),
            nn.SiLU(inplace=True),
            nn.Linear(MEMORY_DIM // 2, 1)
        )
        self.offset_scale = 4.0
        self.offset_head = nn.Sequential(
            nn.Conv2d(
                192,
                96,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                96,
                2,
                kernel_size=1,
                stride=1,
                bias=True
            )
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def forward(self, appearance, geometry, score_map):
        B, C, H, W = appearance.shape

        peak_map = F.max_pool2d(
            score_map,
            kernel_size=3,
            stride=1,
            padding=1
        )
        peak_mask = (score_map >= peak_map).float()
        suppressed_score = score_map * peak_mask

        app = appearance.flatten(2).transpose(1, 2)
        geo = geometry.flatten(2).transpose(1, 2)
        score_logits = suppressed_score.flatten(2).squeeze(1)

        K = min(self.topk, score_logits.shape[1])
        grid_h = int(math.ceil(math.sqrt(K * H / W)))
        grid_w = int(math.ceil(K / grid_h))

        while grid_h * grid_w < K:
            grid_w += 1

        score_grid = score_logits.view(B, H, W)
        selected_indices = []
        selected_scores = []

        for gy in range(grid_h):
            y0 = (gy * H) // grid_h
            y1 = ((gy + 1) * H) // grid_h

            if y1 <= y0:
                continue

            for gx in range(grid_w):
                x0 = (gx * W) // grid_w
                x1 = ((gx + 1) * W) // grid_w

                if x1 <= x0:
                    continue

                region = score_grid[:, y0:y1, x0:x1]
                region_flat = region.reshape(B, -1)
                best_score, best_idx = region_flat.max(dim=1)

                region_width = x1 - x0
                local_y = best_idx // region_width
                local_x = best_idx % region_width

                global_y = y0 + local_y
                global_x = x0 + local_x
                global_idx = global_y * W + global_x

                selected_indices.append(global_idx)
                selected_scores.append(best_score)

        indices = torch.stack(selected_indices, dim=1)
        proposal_logits = torch.stack(selected_scores, dim=1)

        if indices.shape[1] > K:
            proposal_logits, keep = torch.topk(
                proposal_logits,
                k=K,
                dim=1
            )
            indices = torch.gather(indices, 1, keep)

        app = torch.gather(
            app,
            1,
            indices.unsqueeze(-1).expand(-1, -1, app.shape[-1])
        )

        geo = torch.gather(
            geo,
            1,
            indices.unsqueeze(-1).expand(-1, -1, geo.shape[-1])
        )

        proposal_tokens = torch.cat([app, geo], dim=-1)
        proposal_tokens = self.token_projection(proposal_tokens)

        proposal_score_logits = self.score_head(
            proposal_tokens
        ).squeeze(-1)
        proposal_scores = torch.sigmoid(proposal_score_logits)

        y = indices // W
        x = indices % W
        base_centers = torch.stack([x.float(), y.float()], dim=-1)

        offset_map = self.offset_head(appearance)
        offsets = offset_map.flatten(2).transpose(1, 2)
        offsets = torch.gather(
            offsets,
            1,
            indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        offsets = torch.tanh(offsets) * self.offset_scale
        centers = base_centers + offsets

        centers_x = centers[..., 0].clamp(0, W - 1)
        centers_y = centers[..., 1].clamp(0, H - 1)
        centers = torch.stack([centers_x, centers_y], dim=-1)

        return proposal_tokens, proposal_scores, centers, base_centers


class AdaptiveTextProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ProposalEncoder()
        self.center_head = CenterHeatmapHead()
        self.boundary_head = BoundaryHead()
        self.boundary_feature = BoundaryFeatureEncoder()
        self.app_head = AppearanceEmbedding()
        self.geo_head = GeometryEmbedding()
        self.tokenizer = ProposalTokenGenerator(
            token_dim=224,
            topk=100
        )

    def forward(self, shared_quality):
        x = shared_quality["quality_feature"]
        proposal_feature = self.encoder(x)

        center_heatmap = self.center_head(proposal_feature)
        boundary_score = self.boundary_head(proposal_feature)
        boundary_feature = self.boundary_feature(proposal_feature)
        appearance = self.app_head(proposal_feature)
        geometry = self.geo_head(boundary_feature)

        proposal_tokens, proposal_scores, proposal_centers, base_centers = self.tokenizer(
            appearance,
            geometry,
            center_heatmap
        )

        return {
            "proposal_feature": proposal_feature,
            "appearance_feature": appearance,
            "geometry_feature": geometry,
            "proposal_tokens": proposal_tokens,
            "proposal_scores": proposal_scores,
            "proposal_centers": proposal_centers,
            "base_centers": base_centers,
            "boundary_score": boundary_score,
            "center_heatmap": center_heatmap
        }

class DynamicBoundaryMemory(nn.Module):
    def __init__(
        self,
        token_dim=MEMORY_DIM,
        num_memory_slots=16,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.token_dim = token_dim
        self.num_memory_slots = num_memory_slots
        self.num_heads = num_heads

        self.memory_bank = nn.Parameter(
            torch.empty(num_memory_slots, token_dim)
        )
        nn.init.xavier_uniform_(self.memory_bank)

        self.context_encoder = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim)
        )

        self.memory_offset = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(
                token_dim,
                num_memory_slots * token_dim
            )
        )

        self.query_projection = nn.Linear(
            token_dim,
            token_dim
        )
        self.key_projection = nn.Linear(
            token_dim,
            token_dim
        )
        self.value_projection = nn.Linear(
            token_dim,
            token_dim
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)

        self.memory_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid()
        )

        self.memory_update = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim)
        )

        self.memory_norm = nn.LayerNorm(token_dim)

        self.memory_score_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
            nn.Sigmoid()
        )

        self.quality_projection = nn.Sequential(
            nn.Linear(192, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim)
        )

    def forward(self, proposal, quality):
        proposal_tokens = proposal["proposal_tokens"]
        quality_feature = quality["quality_feature"]

        quality_feature = F.adaptive_avg_pool2d(
            quality_feature,
            output_size=1
        ).flatten(1)

        quality_feature = self.quality_projection(
            quality_feature
        )

        B, N, D = proposal_tokens.shape

        global_proposal = proposal_tokens.mean(dim=1)

        context = torch.cat(
            [global_proposal, quality_feature],
            dim=-1
        )
        context = self.context_encoder(context)

        memory_offset = self.memory_offset(context)
        memory_offset = memory_offset.view(
            B,
            self.num_memory_slots,
            self.token_dim
        )

        adaptive_memory = (
            self.memory_bank.unsqueeze(0) + memory_offset
        )

        query = self.query_projection(proposal_tokens)
        key = self.key_projection(adaptive_memory)
        value = self.value_projection(adaptive_memory)

        attended_memory, attention_weights = self.cross_attention(
            query,
            key,
            value,
            need_weights=True
        )

        attended_memory = self.attention_dropout(
            attended_memory
        )

        gate_input = torch.cat(
            [proposal_tokens, attended_memory],
            dim=-1
        )
        memory_gate = self.memory_gate(gate_input)

        evolved_memory = (
            memory_gate * attended_memory
            + (1.0 - memory_gate) * proposal_tokens
        )

        updated_tokens = self.memory_update(evolved_memory)

        memory_tokens = proposal_tokens + updated_tokens
        memory_tokens = self.memory_norm(memory_tokens)

        memory_scores = self.memory_score_head(memory_tokens)

        return {
            "memory_tokens": memory_tokens,
            "memory_scores": memory_scores,
            "adaptive_memory": adaptive_memory,
            "attention_weights": attention_weights,
            "memory_gate": memory_gate,
            "scene_context": context
        }

INSTANCE_DIM = 128


class DetectionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MEMORY_DIM, MEMORY_DIM)
        self.fc2 = nn.Linear(MEMORY_DIM, MEMORY_DIM)
        self.norm = nn.LayerNorm(MEMORY_DIM)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        return self.norm(x)


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = DetectionBlock()

        self.boundary_head = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(MEMORY_DIM, MEMORY_DIM)
        )

        self.objectness_head = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(MEMORY_DIM // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(MEMORY_DIM // 2, 1)
        )

        self.embedding_head = nn.Linear(
            MEMORY_DIM,
            INSTANCE_DIM
        )

    def forward(self, tokens):
        det = self.shared(tokens)
        boundary_representation = self.boundary_head(det)
        objectness = self.objectness_head(det)
        confidence = self.confidence_head(det)

        embedding = F.normalize(
            self.embedding_head(det),
            dim=-1
        )

        return {
            "detection_tokens": det,
            "boundary_representation": boundary_representation,
            "objectness": objectness,
            "confidence": confidence,
            "instance_embedding": embedding
        }




class UnifiedBoundaryEvolutionNetwork(nn.Module):
    def __init__(self, max_vertices=32):
        super().__init__()
        self.max_vertices = max_vertices

        self.context_encoder = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM, bias=False),
            nn.LayerNorm(MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM, bias=False),
            nn.LayerNorm(MEMORY_DIM),
            nn.GELU()
        )

        self.boundary_backbone = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.LayerNorm(MEMORY_DIM)
        )

        self.geometry_branch = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, MEMORY_DIM)
        )

        self.reliability_branch = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, MEMORY_DIM)
        )

        self.boundary_fusion = nn.Sequential(
            nn.Linear(MEMORY_DIM * 2, MEMORY_DIM),
            nn.LayerNorm(MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM)
        )

        self.evolution_gate = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.Sigmoid()
        )

        self.boundary_refinement = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM)
        )

        self.refinement_norm = nn.LayerNorm(MEMORY_DIM)

        self.contour_reconstruction = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, 5)
        )

        self.vertex_start = nn.Parameter(
            torch.empty(1, 1, MEMORY_DIM)
        )
        nn.init.normal_(self.vertex_start, std=0.02)

        self.vertex_input_projection = nn.Sequential(
            nn.Linear(2, MEMORY_DIM),
            nn.LayerNorm(MEMORY_DIM),
            nn.GELU()
        )

        self.vertex_decoder = nn.GRUCell(
            MEMORY_DIM,
            MEMORY_DIM
        )

        self.vertex_offset_head = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, 2)
        )

        self.vertex_stop_head = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, 1)
        )

        self.polygon_confidence = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, detection_output, proposal_output):
        boundary = detection_output["boundary_representation"]

        context = self.context_encoder(boundary)
        context = context + boundary

        evolution = self.boundary_backbone(context)
        evolution = evolution + context

        geometry = self.geometry_branch(evolution)
        reliability = self.reliability_branch(evolution)

        boundary_field = self.boundary_fusion(
            torch.cat([geometry, reliability], dim=-1)
        )

        gate = self.evolution_gate(boundary_field)

        boundary_field = (
            gate * boundary_field
            + (1.0 - gate) * evolution
        )

        refined_boundary = self.boundary_refinement(boundary_field)
        refined_boundary = self.refinement_norm(
            refined_boundary + boundary_field
        )

        geometry_output = self.contour_reconstruction(refined_boundary)

        delta_xy = geometry_output[..., :2]
        curvature = torch.tanh(geometry_output[..., 2:3])
        thickness = F.softplus(geometry_output[..., 3:4])
        boundary_reliability = torch.sigmoid(
            geometry_output[..., 4:5]
        )

        boundary_offsets = delta_xy * boundary_reliability
        boundary_curvature = curvature * boundary_reliability
        boundary_thickness = thickness * boundary_reliability

        continuous_boundary_field = torch.cat(
            [
                boundary_offsets,
                boundary_curvature,
                boundary_thickness,
                boundary_reliability
            ],
            dim=-1
        )

        proposal_centers = proposal_output["proposal_centers"]
        batch_size, num_proposals, _ = proposal_centers.shape

        feature_height, feature_width = (
            proposal_output["boundary_score"].shape[-2:]
        )

        hidden = refined_boundary.reshape(
            batch_size * num_proposals,
            MEMORY_DIM
        )

        hidden = hidden.contiguous()

        previous_vertex = self.vertex_start.expand(
            batch_size,
            num_proposals,
            -1
        ).reshape(batch_size * num_proposals, MEMORY_DIM)

        center_flat = proposal_centers.reshape(
            batch_size * num_proposals,
            2
        )

        vertex_outputs = []
        stop_outputs = []

        offset_scale = center_flat.new_tensor(
            [feature_width / 2.0, feature_height / 2.0]
        )

        for _ in range(self.max_vertices):
            hidden = self.vertex_decoder(previous_vertex, hidden)

            normalized_offset = torch.tanh(
                self.vertex_offset_head(hidden)
            )

            vertex_offset = normalized_offset * offset_scale
            vertex = center_flat + vertex_offset

            vertex_x = vertex[:, 0].clamp(0, feature_width - 1)
            vertex_y = vertex[:, 1].clamp(0, feature_height - 1)
            vertex = torch.stack([vertex_x, vertex_y], dim=-1)

            vertex_outputs.append(vertex.reshape(
                batch_size, num_proposals, 2
            ))

            stop_outputs.append(
                self.vertex_stop_head(hidden).reshape(
                    batch_size, num_proposals
                )
            )

            previous_vertex = self.vertex_input_projection(
                normalized_offset
            )

        polygon = torch.stack(vertex_outputs, dim=2)
        vertex_stop_logits = torch.stack(stop_outputs, dim=2)

        polygon_confidence = self.polygon_confidence(
            refined_boundary
        )

        return {
            "boundary_field": continuous_boundary_field,
            "boundary_offsets": boundary_offsets,
            "boundary_curvature": boundary_curvature,
            "boundary_thickness": boundary_thickness,
            "boundary_reliability": boundary_reliability,
            "polygon": polygon,
            "vertex_stop_logits": vertex_stop_logits,
            "polygon_confidence": polygon_confidence
        }




class DetectionFeatureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.LayerNorm(MEMORY_DIM)
        )

    def forward(self, x):
        if x.dim() == 5 and x.shape[1] == FEATURE_DIM:
            B, C, V, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(B, V * H * W, C)
        elif x.dim() == 5 and x.shape[2] == FEATURE_DIM:
            B, V, C, H, W = x.shape
            x = x.permute(0, 1, 3, 4, 2)
            x = x.reshape(B, V * H * W, C)
        elif x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Unsupported feature shape {x.shape}")

        return self.projection(x)


class ProposalProjection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


