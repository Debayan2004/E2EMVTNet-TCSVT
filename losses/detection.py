import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.float().flatten(1)
        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BalancedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        target = target.float()
        positive = target.sum()
        negative = target.numel() - positive
        pos_weight = negative / (positive + 1e-6)
        return F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=pos_weight.detach()
        )


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = BalancedBCELoss()

    def forward(self, pred, target):
        return (self.dice(pred, target) + self.bce(pred, target)) * 0.5


class ProposalScoreLoss(nn.Module):
    def __init__(self, max_pos_weight=20.0):
        super().__init__()
        self.max_pos_weight = max_pos_weight

    def forward(self, proposal_output, target_scores):
        pred = proposal_output["proposal_scores"].float()
        target = target_scores.float()
        pred = pred.clamp(min=1e-6, max=1.0 - 1e-6)

        positive = target > 0.5
        negative = ~positive
        num_positive = positive.sum()
        num_negative = negative.sum()

        if num_positive.item() > 0:
            pos_weight = num_negative.float() / num_positive.float()
            pos_weight = torch.clamp(
                pos_weight,
                min=1.0,
                max=self.max_pos_weight
            )
        else:
            pos_weight = torch.tensor(1.0, device=pred.device)

        with torch.autocast(device_type="cuda", enabled=False):
            bce = F.binary_cross_entropy(
                pred.float(),
                target.float(),
                reduction="none"
            )

        weights = torch.ones_like(target)
        weights = torch.where(positive, pos_weight, weights)
        return (bce * weights).mean()



class ProposalCenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, proposal_output, target_centers, proposal_targets):
        pred = proposal_output["proposal_centers"]
        positive = proposal_targets > 0.5

        if not positive.any():
            return pred.sum() * 0.0

        pred_positive = pred[positive]
        target_positive = target_centers[positive]

        return F.smooth_l1_loss(pred_positive, target_positive)


class BoundaryHeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = DiceBCELoss()

    def forward(self, proposal_output, target_heatmap):
        return self.loss(
            proposal_output["boundary_score"],
            target_heatmap
        )


class CenterHeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = DiceBCELoss()

    def forward(self, proposal_output, target_center):
        return self.loss(
            proposal_output["center_heatmap"],
            target_center
        )


class MemoryDiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, memory_output):
        tokens = memory_output["memory_tokens"]
        tokens = F.normalize(tokens, dim=-1)
        similarity = torch.matmul(tokens, tokens.transpose(1, 2))
        identity = torch.eye(
            similarity.size(-1),
            device=similarity.device
        )
        return (similarity - identity).pow(2).mean()


class MemoryGateLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, memory_output):
        gate = memory_output["memory_gate"]
        gate = gate.clamp(min=1e-6, max=1.0 - 1e-6)
        entropy = -(
            gate * torch.log(gate)
            + (1 - gate) * torch.log(1 - gate)
        )
        return entropy.mean()


class BoundaryReliabilityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, boundary_output, detection_output):
        boundary = boundary_output["boundary_reliability"].float()
        confidence = detection_output["confidence"].detach().float()

        if confidence.ndim == boundary.ndim - 1:
            confidence = confidence.unsqueeze(-1)
        elif confidence.ndim > boundary.ndim:
            confidence = confidence.squeeze(-1)

        confidence = torch.sigmoid(confidence)

        if confidence.shape != boundary.shape:
            raise ValueError(
                f"Reliability shape mismatch: "
                f"boundary={tuple(boundary.shape)}, "
                f"confidence={tuple(confidence.shape)}"
            )

        return F.mse_loss(boundary, confidence)


class PolygonConfidenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, boundary_output, target_score):
        pred = boundary_output["polygon_confidence"]

        if target_score.dim() == 2:
            target_score = target_score.unsqueeze(-1)

        return F.binary_cross_entropy(
            pred.clamp(min=1e-6, max=1.0 - 1e-6),
            target_score.float()
        )



class BoundarySmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, boundary_output, targets):
        polygon = boundary_output["polygon"].float()
        mask = targets["polygon_vertex_mask"].float()

        if polygon.ndim != 4 or polygon.shape[-1] != 2:
            raise ValueError(
                f"Expected polygon shape [B, N, V, 2], got {polygon.shape}"
            )

        if mask.shape != polygon.shape[:-1]:
            raise ValueError(
                f"Polygon mask shape mismatch: mask={mask.shape}, polygon={polygon.shape}"
            )

        if polygon.shape[2] < 3:
            return polygon.sum() * 0.0

        previous = torch.roll(polygon, shifts=1, dims=2)
        following = torch.roll(polygon, shifts=-1, dims=2)

        second_difference = (
            previous - 2.0 * polygon + following
        ).abs().mean(dim=-1)

        valid_triplets = (
            torch.roll(mask, shifts=1, dims=2)
            * mask
            * torch.roll(mask, shifts=-1, dims=2)
        )

        return (
            second_difference * valid_triplets
        ).sum() / valid_triplets.sum().clamp_min(1.0)



class PolygonVertexLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, boundary_output, targets):
        predicted = boundary_output["polygon"].float()
        target = targets["polygon_vertices"].float()
        mask = targets["polygon_vertex_mask"].float()

        if predicted.shape != target.shape:
            raise ValueError(
                f"Polygon shape mismatch: predicted={predicted.shape}, target={target.shape}"
            )

        if mask.shape != predicted.shape[:-1]:
            raise ValueError(
                f"Polygon mask shape mismatch: mask={mask.shape}, predicted={predicted.shape}"
            )

        distances = F.smooth_l1_loss(
            predicted,
            target,
            reduction="none"
        ).mean(dim=-1)

        return (distances * mask).sum() / mask.sum().clamp_min(1.0)


class VertexStopLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, boundary_output, targets):
        logits = boundary_output["vertex_stop_logits"].float()
        target = targets["polygon_stop_targets"].float()
        mask = targets["polygon_vertex_mask"].float()

        if logits.shape != target.shape or mask.shape != target.shape:
            raise ValueError(
                f"Vertex-stop shape mismatch: logits={logits.shape}, target={target.shape}, mask={mask.shape}"
            )

        losses = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none"
        )

        return (losses * mask).sum() / mask.sum().clamp_min(1.0)


class DetectionBranchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.proposal = ProposalScoreLoss()
        self.proposal_center = ProposalCenterLoss()
        self.boundary = BoundaryHeatmapLoss()
        self.center_heatmap = CenterHeatmapLoss()
        self.memory = MemoryDiversityLoss()
        self.memory_gate = MemoryGateLoss()
        self.reliability = BoundaryReliabilityLoss()
        self.polygon = PolygonConfidenceLoss()
        self.smooth = BoundarySmoothnessLoss()
        self.vertex = PolygonVertexLoss()
        self.vertex_stop = VertexStopLoss()

    def forward(
        self,
        proposal_output,
        memory_output,
        detection_output,
        boundary_output,
        targets
    ):
        losses = {}

        losses["proposal_loss"] = self.proposal(
            proposal_output,
            targets["proposal_scores"]
        )

        losses["proposal_center_loss"] = self.proposal_center(
            proposal_output,
            targets["proposal_centers"],
            targets["proposal_scores"]
        )

        losses["boundary_heatmap_loss"] = self.boundary(
            proposal_output,
            targets["boundary_heatmap"]
        )

        losses["center_heatmap_loss"] = self.center_heatmap(
            proposal_output,
            targets["center_heatmap"]
        )

        losses["memory_diversity_loss"] = self.memory(
            memory_output
        )

        losses["memory_gate_loss"] = self.memory_gate(
            memory_output
        )

        losses["boundary_reliability_loss"] = self.reliability(
            boundary_output,
            detection_output
        )

        losses["polygon_confidence_loss"] = self.polygon(
            boundary_output,
            targets["polygon_confidence"]
        )

        losses["boundary_smoothness_loss"] = self.smooth(
            boundary_output
        )
        losses["polygon_vertex_loss"] = self.vertex(
            boundary_output,
            targets
        )

        losses["vertex_stop_loss"] = self.vertex_stop(
            boundary_output,
            targets
        )

        losses["boundary_smoothness_loss"] = self.smooth(
            boundary_output,
            targets
        )

        return losses