import torch
import torch.nn as nn


class LearnableUncertainty(nn.Module):
    def __init__(self, initial_value=0.0):
        super().__init__()
        self.log_variance = nn.Parameter(
            torch.tensor(float(initial_value))
        )

    def forward(self):
        variance = torch.exp(self.log_variance)
        return {
            "variance": variance,
            "log_variance": self.log_variance
        }


class LossUncertaintyEstimator(nn.Module):
    def __init__(self, loss_names):
        super().__init__()
        self.loss_names = loss_names
        self.estimators = nn.ModuleDict({
            name: LearnableUncertainty()
            for name in loss_names
        })

    def forward(self):
        output = {}
        for name in self.loss_names:
            output[name] = self.estimators[name]()
        return output


class OptimizationStateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

    def forward(
        self,
        performance_output,
        detection_output,
        recognition_output,
        disagreement_output,
        convergence_output
    ):
        det_rel = performance_output["detection_reliability"]
        rec_rel = performance_output["recognition_reliability"]
        det_conf = detection_output["confidence"].mean(1, keepdim=True)
        rec_conf = recognition_output["character_confidence"].mean(
            1,
            keepdim=True
        )
        disagreement = disagreement_output["disagreement_score"]
        continue_prob = convergence_output["continue_probability"]

        state = torch.cat(
            [
                det_rel,
                rec_rel,
                det_conf,
                rec_conf,
                disagreement,
                continue_prob
            ],
            dim=1
        )

        return self.encoder(state)


class DynamicLossController(nn.Module):
    def __init__(self, loss_names):
        super().__init__()
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        self.state_encoder = OptimizationStateEncoder()
        self.predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, self.num_losses)
        )

    def forward(
        self,
        performance_output,
        detection_output,
        recognition_output,
        disagreement_output,
        convergence_output
    ):
        state = self.state_encoder(
            performance_output,
            detection_output,
            recognition_output,
            disagreement_output,
            convergence_output
        )

        scores = self.predictor(state)
        weights = torch.softmax(scores, dim=1)

        output = {}
        for i, name in enumerate(self.loss_names):
            output[name] = weights[:, i:i + 1]

        output["loss_weights"] = weights
        output["optimization_state"] = state

        return output


class DynamicAdaptiveLossRegulator(nn.Module):
    def __init__(self, loss_names):
        super().__init__()
        self.loss_names = loss_names
        self.uncertainty = LossUncertaintyEstimator(loss_names)
        self.controller = DynamicLossController(loss_names)

    def forward(
        self,
        losses,
        performance_output,
        detection_output,
        recognition_output,
        disagreement_output,
        convergence_output
    ):
        uncertainty_output = self.uncertainty()

        controller_output = self.controller(
            performance_output,
            detection_output,
            recognition_output,
            disagreement_output,
            convergence_output
        )

        total_loss = 0
        weighted_losses = {}

        for name in self.loss_names:
            loss = losses[name]
            alpha = controller_output[name]
            sigma = uncertainty_output[name]["log_variance"]

            task_loss = alpha * (
                torch.exp(-sigma) * loss + sigma
            )

            weighted_losses[name] = task_loss.mean()
            total_loss = total_loss + weighted_losses[name]

        return {
            "total_loss": total_loss,
            "weighted_losses": weighted_losses,
            "loss_weights": controller_output["loss_weights"],
            "optimization_state": controller_output["optimization_state"],
            "task_uncertainty": uncertainty_output
        }