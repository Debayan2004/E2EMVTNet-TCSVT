
from engine.metrics import evaluate_umvtsnet


def evaluate_model(model, loader, device):
    return evaluate_umvtsnet(
        model=model,
        loader=loader,
        device=device
    )
