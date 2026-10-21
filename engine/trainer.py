
import torch
from tqdm.auto import tqdm

from data.targets import build_proposal_targets
from data.targets import build_proposal_targets, build_polygon_targets

ACCUMULATION_STEPS = 8
USE_AMP = torch.cuda.is_available()


def prepare_batch(batch, device):
    for key in list(batch.keys()):
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device, non_blocking=True)
    return batch




def prepare_targets(outputs, batch):
    proposal_scores, proposal_centers, matched_gt_indices = (
        build_proposal_targets(
            outputs,
            batch,
            match_radius=8.0,
            return_matches=True
        )
    )

    batch_size, num_proposals = proposal_scores.shape
    max_vertices = 32
    device = proposal_scores.device

    polygon_vertices = torch.zeros(
        batch_size,
        num_proposals,
        max_vertices,
        2,
        dtype=torch.float32,
        device=device
    )

    polygon_vertex_mask = torch.zeros(
        batch_size,
        num_proposals,
        max_vertices,
        dtype=torch.float32,
        device=device
    )

    polygon_stop_targets = torch.zeros(
        batch_size,
        num_proposals,
        max_vertices,
        dtype=torch.float32,
        device=device
    )

    polygon_lengths = torch.zeros(
        batch_size,
        num_proposals,
        dtype=torch.long,
        device=device
    )

    for b in range(batch_size):
        polygon_targets = build_polygon_targets(
            batch["polygons"][b],
            ignore_flags=batch["ignore_flags"][b],
            output_height=outputs["proposal_output"]["boundary_score"].shape[-2],
            output_width=outputs["proposal_output"]["boundary_score"].shape[-1],
            image_height=720,
            image_width=1280,
            max_vertices=max_vertices
        )

        original_indices = torch.as_tensor(
            polygon_targets["original_indices"],
            dtype=torch.long,
            device=device
        )

        vertices = torch.as_tensor(
            polygon_targets["vertices"],
            dtype=torch.float32,
            device=device
        )

        vertex_mask = torch.as_tensor(
            polygon_targets["vertex_mask"],
            dtype=torch.float32,
            device=device
        )

        stop_targets = torch.as_tensor(
            polygon_targets["stop_targets"],
            dtype=torch.float32,
            device=device
        )

        lengths = torch.as_tensor(
            polygon_targets["lengths"],
            dtype=torch.long,
            device=device
        )

        index_to_target = {
            int(original_indices[j].item()): j
            for j in range(original_indices.numel())
        }

        for n in range(num_proposals):
            gt_index = int(matched_gt_indices[b, n].item())

            if gt_index < 0 or proposal_scores[b, n] <= 0.5:
                continue

            target_index = index_to_target.get(gt_index)

            if target_index is None:
                continue

            polygon_vertices[b, n] = vertices[target_index]
            polygon_vertex_mask[b, n] = vertex_mask[target_index]
            polygon_stop_targets[b, n] = stop_targets[target_index]
            polygon_lengths[b, n] = lengths[target_index]

    batch["proposal_scores"] = proposal_scores
    batch["proposal_centers"] = proposal_centers
    batch["matched_gt_indices"] = matched_gt_indices
    batch["polygon_vertices"] = polygon_vertices
    batch["polygon_vertex_mask"] = polygon_vertex_mask
    batch["polygon_stop_targets"] = polygon_stop_targets
    batch["polygon_lengths"] = polygon_lengths
    batch["polygon_confidence"] = proposal_scores.unsqueeze(-1)
    batch["tokens"] = batch["target_tokens"]

    return batch




def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    ema,
    device,
    epoch,
    grad_clip=1.0
):
    model.train()
    criterion.train()

    running = {name: 0.0 for name in criterion.loss_names}
    running["total_loss"] = 0.0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch = prepare_batch(batch, device)
        images = batch["images"]
        input_tokens = batch["input_tokens"]

        amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=USE_AMP
        ):
            outputs = model(images, input_tokens)
            batch = prepare_targets(outputs, batch)
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["total_loss"]

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss / ACCUMULATION_STEPS).backward()

        is_update_step = (batch_idx + 1) % ACCUMULATION_STEPS == 0
        is_last_batch = batch_idx + 1 == len(train_loader)

        if is_update_step or is_last_batch:
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)
        else:
            grad_norm = torch.tensor(0.0, device=device)

        running["total_loss"] += loss.detach().item()

        for name in criterion.loss_names:
            if name in loss_dict["weighted_losses"]:
                running[name] += loss_dict["weighted_losses"][name].detach().item()

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Grad": f"{float(grad_norm):.2f}"
        })

    num_batches = max(1, len(train_loader))

    for key in running:
        running[key] /= num_batches

    if scheduler is not None:
        scheduler.step()

    return running


@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    epoch
):
    model.eval()
    criterion.eval()

    running_total = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False)

    for batch in pbar:
        batch = prepare_batch(batch, device)
        images = batch["images"]
        input_tokens = batch["input_tokens"]

        outputs = model(images, input_tokens)
        batch = prepare_targets(outputs, batch)
        loss_dict = criterion(outputs, batch)
        loss = loss_dict["total_loss"]

        if torch.isfinite(loss):
            running_total += loss.item()
            num_batches += 1

    val_loss = running_total / max(1, num_batches)

    print(f"Validation Epoch {epoch}: loss={val_loss:.6f}")

    return val_loss
