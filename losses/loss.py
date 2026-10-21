import torch.nn as nn

from losses.shared import SharedFeatureLearningLoss
from losses.detection import DetectionBranchLoss
from losses.recognition import RecognitionFeedbackLoss
from losses.regulator import DynamicAdaptiveLossRegulator


class UMVTSLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_loss = SharedFeatureLearningLoss()
        self.det_loss = DetectionBranchLoss()
        self.rec_loss = RecognitionFeedbackLoss()

        self.loss_names = [
            "cross_view_loss",
            "shared_feature_loss",
            "correspondence_loss",
            "pair_attention_loss",
            "reliability_loss",
            "view_entropy_loss",
            "proposal_loss",
            "proposal_center_loss",
            "boundary_heatmap_loss",
            "center_heatmap_loss",
            "memory_diversity_loss",
            "memory_gate_loss",
            "boundary_reliability_loss",
            "polygon_confidence_loss",
            "boundary_smoothness_loss",
            "polygon_vertex_loss",
            "vertex_stop_loss",
            "character_loss",
            "confidence_loss",
            "memory_loss",
            "interaction_loss",
            "feedback_mask_loss",
            "disagreement_loss",
            "convergence_loss",
            "refinement_loss",
            "exchange_loss"
        ]

        self.regulator = DynamicAdaptiveLossRegulator(
            self.loss_names
        )

    def forward(self, outputs, targets):
        shared_losses = self.shared_loss(
            outputs["fusion_output"],
            outputs["correspondence_output"],
            outputs["graph_output"],
            outputs["quality_output"]
        )

        det_losses = self.det_loss(
            outputs["proposal_output"],
            outputs["memory_output"],
            outputs["detection_output"],
            outputs["boundary_output"],
            targets
        )

        rec_losses = self.rec_loss(
            outputs["recognition_output"],
            outputs["recognition_memory"],
            outputs["feedback_output"],
            outputs["exchange_output"],
            outputs["disagreement_output"],
            outputs["convergence_output"],
            outputs["refinement_output"],
            targets["tokens"]
        )

        losses = {}
        losses.update(shared_losses)
        losses.update(det_losses)
        losses.update(rec_losses)

        final_loss = self.regulator(
            losses,
            outputs["performance_output"],
            outputs["detection_output"],
            outputs["recognition_output"],
            outputs["disagreement_output"],
            outputs["convergence_output"]
        )

        return {
            "total_loss": final_loss["total_loss"],
            "raw_losses": losses,
            "weighted_losses": final_loss["weighted_losses"],
            "loss_weights": final_loss["loss_weights"],
            "optimization_state": final_loss["optimization_state"],
            "task_uncertainty": final_loss["task_uncertainty"]
        }