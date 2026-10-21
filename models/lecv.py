
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import FEATURE_DIM
from models.mcre import GN


class PairEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM,
                kernel_size=1,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.embed(x)


class ViewPairGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = PairEmbedding()
        self.pair_index = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3)
        ]

    def forward(self, view_features):
        B, V, C, H, W = view_features.shape
        embedded = []

        for v in range(V):
            feat = self.embed(view_features[:, v])
            embedded.append(feat)

        pairs = []
        for a, b in self.pair_index:
            pair = torch.cat(
                [embedded[a], embedded[b]],
                dim=1
            )
            pairs.append(pair)

        pair_features = torch.stack(pairs, dim=1)

        return {
            "pair_features": pair_features,
            "pair_index": self.pair_index
        }


class PairFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM * 2,
                FEATURE_DIM,
                kernel_size=1,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM,
                kernel_size=3,
                padding=1,
                groups=FEATURE_DIM,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)


class LearnableEpipolarPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PairFeatureEncoder()
        self.direction_head = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM // 2,
                3,
                padding=1,
                bias=False
            ),
            GN(FEATURE_DIM // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                FEATURE_DIM // 2,
                2,
                1
            )
        )
        self.band_head = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM // 2,
                3,
                padding=1,
                bias=False
            ),
            GN(FEATURE_DIM // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                FEATURE_DIM // 2,
                1,
                1
            ),
            nn.Softplus()
        )

    def forward(self, pair_features):
        B, N, C, H, W = pair_features.shape
        encoded = []
        directions = []
        bandwidths = []

        for i in range(N):
            feat = self.encoder(pair_features[:, i])
            direction = self.direction_head(feat)
            direction = F.normalize(direction, dim=1)
            band = self.band_head(feat)

            encoded.append(feat)
            directions.append(direction)
            bandwidths.append(band)

        encoded = torch.stack(encoded, dim=1)
        directions = torch.stack(directions, dim=1)
        bandwidths = torch.stack(bandwidths, dim=1)

        return {
            "pair_embeddings": encoded,
            "epipolar_direction": directions,
            "bandwidth": bandwidths
        }


class OffsetHead(nn.Module):
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
            nn.Conv2d(
                FEATURE_DIM // 2,
                2,
                1
            )
        )

    def forward(self, x):
        return self.head(x)


class ConfidenceHead(nn.Module):
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
            nn.Conv2d(
                FEATURE_DIM // 2,
                1,
                1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)


class RestrictedCorrespondence(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset_head = OffsetHead()
        self.confidence_head = ConfidenceHead()

    def forward(self, prior):
        pair_embeddings = prior["pair_embeddings"]
        directions = prior["epipolar_direction"]
        bandwidth = prior["bandwidth"]

        B, N, C, H, W = pair_embeddings.shape
        aligned = []
        offsets = []
        confidence = []

        for i in range(N):
            feat = pair_embeddings[:, i]
            direction = directions[:, i]
            band = bandwidth[:, i]

            offset = self.offset_head(feat)
            offset = (offset + direction) * band

            conf = self.confidence_head(feat)
            aligned_feat = feat * conf

            aligned.append(aligned_feat)
            offsets.append(offset)
            confidence.append(conf)

        aligned = torch.stack(aligned, dim=1)
        offsets = torch.stack(offsets, dim=1)
        confidence = torch.stack(confidence, dim=1)

        return {
            "aligned_pair_features": aligned,
            "offset_map": offsets,
            "confidence_map": confidence
        }


class FeatureWarper(nn.Module):
    def __init__(self):
        super().__init__()

    def create_base_grid(self, H, W, device):
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        grid = torch.stack([x, y], dim=-1)
        return grid

    def forward(self, feature, offset):
        B, C, H, W = feature.shape

        base = self.create_base_grid(H, W, feature.device)
        base = base.unsqueeze(0).repeat(B, 1, 1, 1)

        dx = offset[:, 0] / (W / 2)
        dy = offset[:, 1] / (H / 2)

        flow = torch.stack([dx, dy], dim=-1)
        grid = base + flow

        warped = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        return warped


class CrossViewRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
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
            GN(FEATURE_DIM)
        )

    def forward(self, x):
        return x + self.refine(x)


class DifferentiableCorrespondence(nn.Module):
    def __init__(self):
        super().__init__()
        self.warper = FeatureWarper()
        self.refine = CrossViewRefinement()

    def forward(self, epipolar_output):
        pair_features = epipolar_output["pair_embeddings"]
        offsets = epipolar_output["epipolar_direction"]
        confidence = epipolar_output["bandwidth"]

        B, N, C, H, W = pair_features.shape
        warped = []
        refined = []

        for i in range(N):
            feat = pair_features[:, i]
            off = offsets[:, i]
            conf = confidence[:, i]

            warp = self.warper(feat, off)
            warp = warp * conf
            refine = self.refine(warp)

            warped.append(warp)
            refined.append(refine)

        warped = torch.stack(warped, dim=1)
        refined = torch.stack(refined, dim=1)

        return {
            "warped_pair_features": warped,
            "refined_pair_features": refined
        }


class PairAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                FEATURE_DIM,
                FEATURE_DIM // 4,
                1,
                bias=False
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                FEATURE_DIM // 4,
                1,
                1
            )
        )

    def forward(self, x):
        return self.score(x)


class CrossViewGraphAggregation(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = PairAttention()
        self.refine = nn.Sequential(
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
            GN(FEATURE_DIM)
        )
        self.pair_map = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3)
        ]

    def forward(self, correspondence):
        pair_feats = correspondence["refined_pair_features"]
        B, N, C, H, W = pair_feats.shape

        pair_scores = []
        for i in range(N):
            score = self.attention(pair_feats[:, i])
            pair_scores.append(score)

        pair_scores = torch.stack(pair_scores, dim=1)
        pair_weights = torch.softmax(pair_scores, dim=1)

        view_features = []
        for view in range(4):
            feat_sum = torch.zeros_like(pair_feats[:, 0])
            weight_sum = torch.zeros_like(pair_weights[:, 0])

            for idx, (a, b) in enumerate(self.pair_map):
                if a == view or b == view:
                    weight = pair_weights[:, idx]
                    feat = pair_feats[:, idx]
                    feat_sum += feat * weight
                    weight_sum += weight

            view_feat = feat_sum / (weight_sum + 1e-6)
            view_feat = view_feat + self.refine(view_feat)
            view_features.append(view_feat)

        view_features = torch.stack(view_features, dim=1)
        shared_feature = view_features.mean(dim=1)

        return {
            "aligned_view_features": view_features,
            "aligned_shared_feature": shared_feature,
            "pair_attention": pair_weights
        }
