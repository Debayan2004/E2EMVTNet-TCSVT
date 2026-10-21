
import torch
import torch.nn as nn
from configs.config import FEATURE_DIM, MEMORY_DIM
from models.mcre import GN


class ProposalProjection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MemoryRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=MEMORY_DIM,
            num_heads=8,
            batch_first=True
        )

    def forward(self, query, memory):
        output, _ = self.attention(query, memory, memory)
        return output


class MemoryGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(MEMORY_DIM * 3, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.Sigmoid()
        )

    def forward(self, token, memory, reliability):
        reliability = reliability.unsqueeze(1)
        reliability = reliability.expand(-1, token.size(1), -1)
        features = torch.cat([token, memory, reliability], dim=-1)
        return self.gate(features)


class ReliabilityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(FEATURE_DIM, MEMORY_DIM, kernel_size=1, bias=False),
            GN(MEMORY_DIM),
            nn.SiLU(inplace=True)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                MEMORY_DIM,
                MEMORY_DIM,
                kernel_size=3,
                padding=1,
                groups=MEMORY_DIM,
                bias=False
            ),
            GN(MEMORY_DIM),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.refine(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class QueryFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=MEMORY_DIM,
            num_heads=8,
            batch_first=True
        )

    def forward(self, tokens, reliability_tokens):
        output, _ = self.cross_attention(
            tokens,
            reliability_tokens,
            reliability_tokens
        )
        return tokens + output


class MemoryRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM)
        )
        self.norm = nn.LayerNorm(MEMORY_DIM)

    def forward(self, x):
        return self.norm(x + self.refine(x))


class DynamicEpisodicMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = ProposalProjection()
        self.read = MemoryRead()
        self.reliability_encoder = ReliabilityEncoder()
        self.query_fusion = QueryFusion()
        self.gate = MemoryGate()
        self.refine = MemoryRefinement()
        self.num_memory_slots = 32
        self.memory_bank = nn.Parameter(
            torch.randn(self.num_memory_slots, MEMORY_DIM)
        )
        nn.init.xavier_uniform_(self.memory_bank)

    def forward(self, proposal_output, quality_output):
        quality_feature = quality_output["quality_feature"]
        tokens = self.project(proposal_output["proposal_tokens"])
        batch_size = tokens.size(0)
        memory = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        reliability_tokens = self.reliability_encoder(quality_feature)
        query = self.query_fusion(tokens, reliability_tokens)
        reliability_global = reliability_tokens.mean(dim=1)
        read_memory = self.read(query, memory)
        gate = self.gate(tokens, read_memory, reliability_global)
        updated_memory = gate * tokens + (1 - gate) * read_memory
        refined_memory = self.refine(updated_memory)

        return {
            "shared_memory_tokens": refined_memory,
            "memory_tokens": updated_memory,
            "memory_gate": gate,
            "memory_read": read_memory,
            "projected_tokens": tokens,
            "reliability_tokens": reliability_tokens,
            "query_tokens": query,
            "scene_token": refined_memory.mean(dim=1)
        }
