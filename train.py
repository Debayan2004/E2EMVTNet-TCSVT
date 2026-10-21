
import copy
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs.config import (
    DATASET_ROOT,
    MAX_TEXT_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
)
from data.dataset import SpottingDataset, spotting_collate_fn
from models.e2emvtnet import E2EMVTNet
from losses.loss import UMVTSLoss
from engine.trainer import train_one_epoch, validate_one_epoch


LEARNING_RATE = 3e-5
LOSS_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
PATIENCE = 20
CHECKPOINT_DIR = "checkpoints"
SEED = 42
USE_AMP = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for key in ema_state:
            if ema_state[key].dtype.is_floating_point:
                ema_state[key].mul_(self.decay).add_(
                    model_state[key],
                    alpha=1.0 - self.decay
                )
            else:
                ema_state[key].copy_(model_state[key])


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    scene_dirs = sorted(
        os.path.join(DATASET_ROOT, name)
        for name in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, name))
    )

    train_scenes, temporary_scenes = train_test_split(
        scene_dirs,
        test_size=0.20,
        random_state=SEED
    )

    val_scenes, _ = train_test_split(
        temporary_scenes,
        test_size=0.50,
        random_state=SEED
    )

    train_dataset = SpottingDataset(
        train_scenes,
        max_text_length=MAX_TEXT_LENGTH
    )
    val_dataset = SpottingDataset(
        val_scenes,
        max_text_length=MAX_TEXT_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=spotting_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=spotting_collate_fn
    )

    model = E2EMVTNet().to(DEVICE)
    criterion = UMVTSLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": LEARNING_RATE},
            {"params": criterion.parameters(), "lr": LOSS_LEARNING_RATE}
        ],
        betas=BETAS,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    ema = ModelEMA(model, EMA_DECAY)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, EPOCHS + 1):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=DEVICE,
            epoch=epoch,
            grad_clip=GRAD_CLIP
        )

        val_loss = validate_one_epoch(
            model=ema.ema,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            epoch=epoch
        )

        history["train_loss"].append(train_stats["total_loss"])
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "history": history
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                checkpoint,
                os.path.join(CHECKPOINT_DIR, "best_e2emvtnet.pt")
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            torch.save(
                checkpoint,
                os.path.join(CHECKPOINT_DIR, f"e2emvtnet_epoch_{epoch}.pt")
            )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_stats['total_loss']:.6f} | "
            f"Validation loss: {val_loss:.6f}"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    torch.save(
        history,
        os.path.join(CHECKPOINT_DIR, "history.pt")
    )


if __name__ == "__main__":
    main()
evaluate.py
