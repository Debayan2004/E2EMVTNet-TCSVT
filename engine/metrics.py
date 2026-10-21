
import numpy as np
import torch

from configs.config import PAD_IDX, EOS_IDX, IDX2CHAR
from data.targets import get_gt_centers


EVAL_PROPOSAL_THRESHOLD = 0.50
CENTER_MATCH_DISTANCE = 8.0


def euclidean_distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def match_centers(predicted_centers, gt_centers, max_distance):
    predicted_centers = [
        np.asarray(x, dtype=np.float32) for x in predicted_centers
    ]
    gt_centers = [
        np.asarray(x, dtype=np.float32) for x in gt_centers
    ]

    if not predicted_centers:
        return 0, 0, len(gt_centers)

    if not gt_centers:
        return 0, len(predicted_centers), 0

    distances = [
        (euclidean_distance(pred, gt), p_idx, g_idx)
        for p_idx, pred in enumerate(predicted_centers)
        for g_idx, gt in enumerate(gt_centers)
    ]

    distances.sort(key=lambda item: item[0])

    matched_pred = set()
    matched_gt = set()

    for distance, p_idx, g_idx in distances:
        if distance > max_distance:
            break

        if p_idx in matched_pred or g_idx in matched_gt:
            continue

        matched_pred.add(p_idx)
        matched_gt.add(g_idx)

    tp = len(matched_pred)
    fp = len(predicted_centers) - tp
    fn = len(gt_centers) - len(matched_gt)

    return tp, fp, fn


def calculate_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def levenshtein_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            substitution = previous[j - 1] + int(char_a != char_b)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1

            current.append(
                min(substitution, insertion, deletion)
            )

        previous = current

    return previous[-1]


def decode_tokens(tokens):
    chars = []

    for token in tokens:
        token = int(token)

        if token == PAD_IDX or token == EOS_IDX:
            break

        character = IDX2CHAR.get(token)

        if character is not None:
            chars.append(character)

    return "".join(chars)


def update_recognition_metrics(prediction, target, stats):
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()

    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Recognition predictions and targets must have shape [B, L]."
        )

    if prediction.shape[0] != target.shape[0]:
        raise ValueError(
            "Recognition predictions and targets must have equal batch sizes."
        )

    for pred_tokens, target_tokens in zip(prediction, target):
        pred_text = decode_tokens(pred_tokens)
        gt_text = decode_tokens(target_tokens)

        stats["total_words"] += 1
        stats["correct_words"] += int(pred_text == gt_text)
        stats["edit_distance"] += levenshtein_distance(
            pred_text, gt_text
        )
        stats["normalization_length"] += max(len(gt_text), 1)


def match_spotting_instances(
    predicted_instances,
    ground_truth_instances,
    max_distance
):
    distances = []

    for p_idx, pred in enumerate(predicted_instances):
        for g_idx, gt in enumerate(ground_truth_instances):
            distance = euclidean_distance(
                pred["center"], gt["center"]
            )

            if distance <= max_distance:
                distances.append((distance, p_idx, g_idx))

    distances.sort(key=lambda item: item[0])

    matched_pred = set()
    matched_gt = set()
    tp = 0

    for _, p_idx, g_idx in distances:
        if p_idx in matched_pred or g_idx in matched_gt:
            continue

        matched_pred.add(p_idx)
        matched_gt.add(g_idx)

        pred_text = predicted_instances[p_idx]["text"]
        gt_text = ground_truth_instances[g_idx]["text"]

        if pred_text == gt_text:
            tp += 1

    fp = len(predicted_instances) - tp
    fn = len(ground_truth_instances) - tp

    return tp, fp, fn


def evaluate_umvtsnet(model, loader, device):
    model.eval()

    det_stats = {
        "tp": 0,
        "fp": 0,
        "fn": 0
    }

    recognition_stats = {
        "correct_words": 0,
        "total_words": 0,
        "edit_distance": 0,
        "normalization_length": 0
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(
                device, non_blocking=True
            )

            input_tokens = batch["input_tokens"].to(
                device, dtype=torch.long, non_blocking=True
            )

            target_tokens = batch["target_tokens"].to(
                device, dtype=torch.long, non_blocking=True
            )

            outputs = model(images, input_tokens)

            proposal_output = outputs["proposal_output"]
            recognition_output = outputs["recognition_output"]

            proposal_scores = proposal_output["proposal_scores"]
            proposal_centers = proposal_output["proposal_centers"]
            boundary_score = proposal_output["boundary_score"]

            for b in range(proposal_scores.shape[0]):
                keep = (
                    proposal_scores[b] >= EVAL_PROPOSAL_THRESHOLD
                )

                pred_centers = (
                    proposal_centers[b][keep]
                    .detach()
                    .cpu()
                    .numpy()
                )

                gt_centers = get_gt_centers(
                    batch["polygons"][b],
                    batch["ignore_flags"][b],
                    image_width=1024.0,
                    image_height=256.0,
                    output_width=boundary_score.shape[-1],
                    output_height=boundary_score.shape[-2]
                )

                tp, fp, fn = match_centers(
                    pred_centers,
                    gt_centers,
                    CENTER_MATCH_DISTANCE
                )

                det_stats["tp"] += tp
                det_stats["fp"] += fp
                det_stats["fn"] += fn

            update_recognition_metrics(
                recognition_output["predicted_indices"],
                target_tokens,
                recognition_stats
            )

    detection_metrics = {
        "matching_method": "center_distance",
        "matching_threshold": CENTER_MATCH_DISTANCE,
        "proposal_threshold": EVAL_PROPOSAL_THRESHOLD,
        "tp": det_stats["tp"],
        "fp": det_stats["fp"],
        "fn": det_stats["fn"],
        **calculate_prf(
            det_stats["tp"],
            det_stats["fp"],
            det_stats["fn"]
        )
    }

    word_accuracy = (
        recognition_stats["correct_words"]
        / max(recognition_stats["total_words"], 1)
    )

    ned = (
        recognition_stats["edit_distance"]
        / max(recognition_stats["normalization_length"], 1)
    )

    recognition_metrics = {
        "word_accuracy": word_accuracy,
        "ned": ned,
        "normalized_edit_score": 1.0 - ned,
        "evaluation_mode": "teacher_forced"
    }

    spotting_metrics = {
        "tp": None,
        "fp": None,
        "fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "status": "requires_per_instance_text_predictions"
    }

    return {
        "detection": detection_metrics,
        "recognition": recognition_metrics,
        "spotting": spotting_metrics
    }
