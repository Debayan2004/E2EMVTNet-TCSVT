
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs.config import (
    DATASET_ROOT,
    MAX_TEXT_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
)
from data.dataset import SpottingDataset, spotting_collate_fn
from models.e2emvtnet import E2EMVTNet
from engine.evaluator import evaluate_model


CHECKPOINT_PATH = "checkpoints/best_e2emvtnet.pt"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    scene_dirs = sorted(
        os.path.join(DATASET_ROOT, name)
        for name in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, name))
    )

    _, temporary_scenes = train_test_split(
        scene_dirs,
        test_size=0.20,
        random_state=SEED
    )

    _, test_scenes = train_test_split(
        temporary_scenes,
        test_size=0.50,
        random_state=SEED
    )

    test_dataset = SpottingDataset(
        test_scenes,
        max_text_length=MAX_TEXT_LENGTH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=spotting_collate_fn
    )

    model = E2EMVTNet().to(DEVICE)

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model_state = checkpoint.get("ema")

    if model_state is None:
        model_state = checkpoint["model"]

    model.load_state_dict(model_state)

    results = evaluate_model(
        model=model,
        loader=test_loader,
        device=DEVICE
    )

    print("Detection metrics")

    for name, value in results["detection"].items():
        print(f"{name}: {value}")

    print("Recognition metrics")

    for name, value in results["recognition"].items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
