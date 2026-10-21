import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph_output):
        views = graph_output["aligned_view_features"]
        mean_feature = views.mean(dim=1, keepdim=True)
        loss = (views - mean_feature).pow(2).mean()
        return loss


class SharedFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fusion_output, graph_output):
        before = fusion_output["shared_feature"]
        after = graph_output["aligned_shared_feature"]
        return F.smooth_l1_loss(before.mean(2), after)


class CorrespondenceSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, correspondence_output):
        feature = correspondence_output["refined_pair_features"]
        dx = feature[:, :, :, :, 1:] - feature[:, :, :, :, :-1]
        dy = feature[:, :, :, 1:, :] - feature[:, :, :, :-1, :]
        return (dx.abs().mean() + dy.abs().mean()) * 0.5


class PairAttentionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph_output):
        attention = graph_output["pair_attention"]
        attention = attention.view(attention.size(0), -1)
        attention = attention.softmax(dim=1)
        target = torch.full_like(attention, 1.0 / attention.size(1))
        return F.kl_div(attention.log(), target, reduction="batchmean")


class ReliabilityCalibrationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, quality_output):
        reliability = quality_output["reliability_map"]
        quality = quality_output["quality_map"]
        return F.mse_loss(reliability, quality.detach())


class ViewWeightEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, quality_output):
        weights = quality_output["view_weights"]
        weights = weights.clamp(min=1e-6, max=1.0)
        entropy = -(weights * torch.log(weights)).sum(dim=1)
        entropy = entropy / math.log(weights.shape[1])
        return entropy.mean()


class SharedFeatureLearningLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_view = CrossViewConsistencyLoss()
        self.shared = SharedFeatureLoss()
        self.correspondence = CorrespondenceSmoothnessLoss()
        self.attention = PairAttentionLoss()
        self.reliability = ReliabilityCalibrationLoss()
        self.view_entropy = ViewWeightEntropyLoss()

    def forward(
        self,
        fusion_output,
        correspondence_output,
        graph_output,
        quality_output
    ):
        losses = {}

        losses["cross_view_loss"] = self.cross_view(graph_output)
        losses["shared_feature_loss"] = self.shared(
            fusion_output,
            graph_output
        )
        losses["correspondence_loss"] = self.correspondence(
            correspondence_output
        )
        losses["pair_attention_loss"] = self.attention(graph_output)
        losses["reliability_loss"] = self.reliability(quality_output)
        losses["view_entropy_loss"] = self.view_entropy(quality_output)
        

        return losses