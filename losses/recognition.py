import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import PAD_IDX


class CharacterRecognitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def forward(self, recognition_output, target_tokens):
        logits = recognition_output["character_logits"]
        B, L, C = logits.shape
        return self.ce(
            logits.reshape(B * L, C),
            target_tokens.reshape(-1)
        )


class CharacterConfidenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recognition_output, target_tokens):
        logits = recognition_output["character_logits"]
        confidence = recognition_output["character_confidence"]
        pred = logits.argmax(dim=-1)
        target = (pred == target_tokens).float()
        confidence = confidence.float().clamp(min=1e-6, max=1.0 - 1e-6)
        mask = (target_tokens != PAD_IDX).float()

        loss = -(
            target * torch.log(confidence)
            + (1 - target) * torch.log(1 - confidence)
        )

        return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class RecognitionMemoryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, memory_output):
        query = memory_output["query_tokens"]
        refined = memory_output["shared_memory_tokens"]
        return F.smooth_l1_loss(refined, query)


class InteractionAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, exchange_output):
        det = exchange_output["updated_detection_tokens"].mean(1)
        rec = exchange_output["updated_recognition_tokens"].mean(1)

        det = F.normalize(det, dim=-1)
        rec = F.normalize(rec, dim=-1)

        cosine = 1 - (det * rec).sum(-1).mean()
        mse = F.mse_loss(det, rec)

        return cosine + 0.5 * mse


class FeedbackMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feedback_output):
        det = feedback_output["detection_feedback_mask"]
        rec = feedback_output["recognition_feedback_mask"]

        det_entropy = -(det * torch.log(det + 1e-6))
        rec_entropy = -(rec * torch.log(rec + 1e-6))
        sparsity = det.mean() + rec.mean()

        return det_entropy.mean() + rec_entropy.mean() + 0.05 * sparsity


class DisagreementLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disagreement_output):
        cosine = disagreement_output["cosine_similarity"]
        disagreement = disagreement_output["disagreement_score"]
        target = 1 - cosine
        return F.mse_loss(disagreement, target.detach())


class ConvergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, convergence_output, disagreement_output):
        continue_prob = convergence_output["continue_probability"]
        target = disagreement_output["disagreement_score"].detach()

        with torch.autocast(device_type="cuda", enabled=False):
            return F.binary_cross_entropy(
                continue_prob.float(),
                target.float()
            )


class RefinementConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, exchange_output, refinement_output):
        det_before = exchange_output["updated_detection_tokens"]
        det_after = refinement_output["refined_detection_tokens"]
        rec_before = exchange_output["updated_recognition_tokens"]
        rec_after = refinement_output["refined_recognition_tokens"]

        det_loss = F.smooth_l1_loss(det_after, det_before.detach())
        rec_loss = F.smooth_l1_loss(rec_after, rec_before.detach())

        return (det_loss + rec_loss) * 0.5


class ExchangeConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, memory_output, exchange_output):
        det_before = memory_output["memory_tokens"].mean(1)
        det_after = exchange_output["updated_detection_tokens"].mean(1)
        rec_before = memory_output["shared_memory_tokens"].mean(1)
        rec_after = exchange_output["updated_recognition_tokens"].mean(1)

        det_loss = F.smooth_l1_loss(det_after, det_before.detach())
        rec_loss = F.smooth_l1_loss(rec_after, rec_before.detach())

        return 0.5 * (det_loss + rec_loss)


class RecognitionFeedbackLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.character = CharacterRecognitionLoss()
        self.confidence = CharacterConfidenceLoss()
        self.memory = RecognitionMemoryLoss()
        self.interaction = InteractionAlignmentLoss()
        self.feedback = FeedbackMaskLoss()
        self.disagreement = DisagreementLoss()
        self.convergence = ConvergenceLoss()
        self.refinement = RefinementConsistencyLoss()
        self.exchange = ExchangeConsistencyLoss()

    def forward(
        self,
        recognition_output,
        memory_output,
        feedback_output,
        exchange_output,
        disagreement_output,
        convergence_output,
        refinement_output,
        target_tokens
    ):
        losses = {}

        losses["character_loss"] = self.character(
            recognition_output,
            target_tokens
        )

        losses["confidence_loss"] = self.confidence(
            recognition_output,
            target_tokens
        )

        losses["memory_loss"] = self.memory(memory_output)

        losses["interaction_loss"] = self.interaction(exchange_output)

        losses["feedback_mask_loss"] = self.feedback(feedback_output)

        losses["disagreement_loss"] = self.disagreement(
            disagreement_output
        )

        losses["convergence_loss"] = self.convergence(
            convergence_output,
            disagreement_output
        )

        losses["refinement_loss"] = self.refinement(
            exchange_output,
            refinement_output
        )

        losses["exchange_loss"] = self.exchange(
            memory_output,
            exchange_output
        )

        return losses