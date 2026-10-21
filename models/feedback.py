import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import MEMORY_DIM


class DetectionReliabilityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(3, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, detection_output):
        confidence = detection_output["confidence"]
        confidence_mean = confidence.mean(dim=1, keepdim=True)
        confidence_std = confidence.std(dim=1, keepdim=True, unbiased=False)
        confidence_max = confidence.max(dim=1, keepdim=True).values
        x = torch.cat(
            [confidence_mean, confidence_std, confidence_max],
            dim=1
        )
        return self.predictor(x)


class RecognitionReliabilityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(3, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, recognition_output):
        confidence = recognition_output["character_confidence"]
        confidence_mean = confidence.mean(dim=1, keepdim=True)
        confidence_std = confidence.std(dim=1, keepdim=True, unbiased=False)
        confidence_min = confidence.min(dim=1, keepdim=True).values
        x = torch.cat(
            [confidence_mean, confidence_std, confidence_min],
            dim=1
        )
        return self.predictor(x)


class BranchPerformanceEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_estimator = DetectionReliabilityEstimator()
        self.rec_estimator = RecognitionReliabilityEstimator()

    def forward(self, detection_output, recognition_output):
        detection_reliability = self.det_estimator(detection_output)
        recognition_reliability = self.rec_estimator(recognition_output)
        return {
            "detection_reliability": detection_reliability,
            "recognition_reliability": recognition_reliability
        }


class AdaptiveFeedbackController(nn.Module):
    def __init__(self):
        super().__init__()
        self.controller = nn.Sequential(
            nn.Linear(2, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, 2)
        )

    def forward(self, performance_output):
        det = performance_output["detection_reliability"]
        rec = performance_output["recognition_reliability"]
        reliability = torch.cat([det, rec], dim=1)
        scores = self.controller(reliability)
        weights = torch.softmax(scores, dim=1)
        return {
            "detection_feedback_weight": weights[:, 0:1],
            "recognition_feedback_weight": weights[:, 1:2],
            "feedback_weights": weights,
            "reliability": reliability
        }


class DetectionTokenProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Identity()

    def forward(self, x):
        return self.project(x)


class TokenReliabilityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predict(x)


class TokenWiseAdaptiveFeedback(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_projection = DetectionTokenProjection()
        self.det_predictor = TokenReliabilityPredictor()
        self.rec_predictor = TokenReliabilityPredictor()

    def forward(
        self,
        detection_output,
        recognition_output,
        controller_output
    ):
        detection_tokens = detection_output["boundary_representation"]
        detection_tokens = self.det_projection(detection_tokens)
        recognition_tokens = recognition_output["recognition_tokens"]

        det_local = self.det_predictor(detection_tokens)
        rec_local = self.rec_predictor(recognition_tokens)

        det_global = controller_output[
            "detection_feedback_weight"
        ].unsqueeze(1)
        rec_global = controller_output[
            "recognition_feedback_weight"
        ].unsqueeze(1)

        detection_mask = det_local * det_global
        recognition_mask = rec_local * rec_global

        return {
            "detection_tokens": detection_tokens,
            "recognition_tokens": recognition_tokens,
            "detection_feedback_mask": detection_mask,
            "recognition_feedback_mask": recognition_mask
        }


class BidirectionalFeatureExchange(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_to_rec = nn.MultiheadAttention(
            embed_dim=MEMORY_DIM,
            num_heads=8,
            batch_first=True
        )
        self.rec_to_det = nn.MultiheadAttention(
            embed_dim=MEMORY_DIM,
            num_heads=8,
            batch_first=True
        )
        self.det_gate = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.Sigmoid()
        )
        self.rec_gate = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.Sigmoid()
        )
        self.det_norm = nn.LayerNorm(MEMORY_DIM)
        self.rec_norm = nn.LayerNorm(MEMORY_DIM)

    def forward(self, feedback_features):
        det = feedback_features["detection_tokens"]
        rec = feedback_features["recognition_tokens"]
        det_mask = feedback_features["detection_feedback_mask"]
        rec_mask = feedback_features["recognition_feedback_mask"]

        rec_message, _ = self.det_to_rec(
            query=rec,
            key=det,
            value=det
        )
        det_message, _ = self.rec_to_det(
            query=det,
            key=rec,
            value=rec
        )

        rec_message = rec_message * rec_mask
        det_message = det_message * det_mask

        rec = rec + self.rec_gate(rec_message) * rec_message
        det = det + self.det_gate(det_message) * det_message

        rec = self.rec_norm(rec)
        det = self.det_norm(det)

        return {
            "updated_detection_tokens": det,
            "updated_recognition_tokens": rec,
            "detection_message": det_message,
            "recognition_message": rec_message
        }


class CollaborativeDisagreementEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_projection = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.LayerNorm(MEMORY_DIM)
        )
        self.rec_projection = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.LayerNorm(MEMORY_DIM)
        )
        self.encoder = nn.Sequential(
            nn.Linear(MEMORY_DIM * 3, MEMORY_DIM),
            nn.GELU(),
            nn.LayerNorm(MEMORY_DIM),
            nn.Linear(MEMORY_DIM, MEMORY_DIM // 2),
            nn.GELU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(MEMORY_DIM // 2 + 1, MEMORY_DIM // 4),
            nn.GELU(),
            nn.Linear(MEMORY_DIM // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, exchange_output):
        det = exchange_output["updated_detection_tokens"]
        rec = exchange_output["updated_recognition_tokens"]

        det_global = self.det_projection(det.mean(dim=1))
        rec_global = self.rec_projection(rec.mean(dim=1))

        absolute_difference = torch.abs(det_global - rec_global)
        elementwise_product = det_global * rec_global
        cosine = F.cosine_similarity(
            det_global,
            rec_global,
            dim=-1
        ).unsqueeze(-1)

        semantic = torch.cat(
            [absolute_difference, elementwise_product, det_global],
            dim=-1
        )
        semantic = self.encoder(semantic)

        disagreement = torch.cat([semantic, 1 - cosine], dim=-1)
        disagreement = self.predictor(disagreement)

        return {
            "detection_summary": det_global,
            "recognition_summary": rec_global,
            "cosine_similarity": cosine,
            "disagreement_score": disagreement
        }


class AdaptiveConvergenceController(nn.Module):
    def __init__(self):
        super().__init__()
        self.controller = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        performance_output,
        feedback_output,
        disagreement_output
    ):
        det_rel = performance_output["detection_reliability"]
        rec_rel = performance_output["recognition_reliability"]
        feedback_weights = feedback_output["feedback_weights"]
        disagreement = disagreement_output["disagreement_score"]

        x = torch.cat(
            [det_rel, rec_rel, feedback_weights, disagreement],
            dim=1
        )
        continue_probability = self.controller(x)
        stop_probability = 1.0 - continue_probability

        return {
            "continue_probability": continue_probability,
            "stop_probability": stop_probability
        }


class RefinementGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(MEMORY_DIM + 1, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.Sigmoid()
        )

    def forward(self, tokens, continue_probability):
        B, N, C = tokens.shape
        probability = continue_probability.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([tokens, probability], dim=-1)
        return self.gate(x)


class CollaborativeRefinementCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_gate = RefinementGate()
        self.rec_gate = RefinementGate()
        self.det_refine = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM)
        )
        self.rec_refine = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, MEMORY_DIM)
        )
        self.det_norm = nn.LayerNorm(MEMORY_DIM)
        self.rec_norm = nn.LayerNorm(MEMORY_DIM)

    def forward(self, exchange_output, convergence_output):
        det = exchange_output["updated_detection_tokens"]
        rec = exchange_output["updated_recognition_tokens"]
        probability = convergence_output["continue_probability"]

        det_gate = self.det_gate(det, probability)
        rec_gate = self.rec_gate(rec, probability)

        det_update = self.det_refine(det)
        rec_update = self.rec_refine(rec)

        det = self.det_norm(det + det_gate * det_update)
        rec = self.rec_norm(rec + rec_gate * rec_update)

        return {
            "refined_detection_tokens": det,
            "refined_recognition_tokens": rec,
            "detection_refinement_gate": det_gate,
            "recognition_refinement_gate": rec_gate,
            "continue_probability": probability,
            "stop_probability": convergence_output["stop_probability"]
        }