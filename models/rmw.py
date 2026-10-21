
import torch
import torch.nn as nn

from configs.config import FEATURE_DIM
from models.mcre import GN


class QualityEncoder(nn.Module):
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


class QualityHead(nn.Module):
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


class MultiViewQualityAssessment(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = QualityEncoder()
        self.reliability = QualityHead()
        self.visibility = QualityHead()
        self.occlusion = QualityHead()
        self.quality = QualityHead()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, graph_output):
        views = graph_output["aligned_view_features"]
        B, V, C, H, W = views.shape

        refined_views = []
        reliability = []
        visibility = []
        occlusion = []
        quality = []

        for i in range(V):
            feat = self.encoder(views[:, i])
            refined_views.append(feat)
            reliability.append(self.reliability(feat))
            visibility.append(self.visibility(feat))
            occlusion.append(self.occlusion(feat))
            quality.append(self.quality(feat))

        refined_views = torch.stack(refined_views, dim=1)
        reliability = torch.stack(reliability, dim=1)
        visibility = torch.stack(visibility, dim=1)
        occlusion = torch.stack(occlusion, dim=1)
        quality = torch.stack(quality, dim=1)

        mean_score = (
            reliability
            + visibility
            + quality
            + (1.0 - occlusion)
        ) / 4.0

        eps = 1e-6
        product_score = (
            reliability
            * visibility
            * quality
            * (1.0 - occlusion)
        ).clamp(min=eps)

        alpha = torch.sigmoid(self.alpha)

        fused_score = (
            alpha * mean_score
            + (1.0 - alpha) * product_score
        )

        view_weights = torch.softmax(fused_score, dim=1)

        weighted_views = (
            refined_views
            * view_weights.expand_as(refined_views)
        )

        quality_feature = weighted_views.sum(dim=1)

        return {
            "refined_view_features": refined_views,
            "reliability_map": reliability,
            "visibility_map": visibility,
            "occlusion_map": occlusion,
            "quality_map": quality,
            "view_weights": view_weights,
            "quality_feature": quality_feature,
            "mean_score": mean_score,
            "product_score": product_score,
            "fusion_alpha": alpha
        }
