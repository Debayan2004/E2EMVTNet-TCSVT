import numpy as np
import torch
import cv2

def get_gt_centers(
    polygons,
    ignore_flags=None,
    image_width=1280.0,
    image_height=720.0,
    output_width=256.0,
    output_height=64.0
):
    centers = []

    scale_x = output_width / image_width
    scale_y = output_height / image_height

    if polygons is None:
        return centers

    for view_idx, view_polygons in enumerate(polygons):
        if view_polygons is None:
            continue

        if ignore_flags is not None and view_idx < len(ignore_flags):
            view_ignore = ignore_flags[view_idx]
        else:
            view_ignore = None

        for poly_idx, polygon in enumerate(view_polygons):
            if view_ignore is not None and poly_idx < len(view_ignore):
                flag = view_ignore[poly_idx]

                if isinstance(flag, (list, tuple, np.ndarray)):
                    flag = np.asarray(flag).reshape(-1)
                    flag = bool(flag[0]) if len(flag) > 0 else False
                else:
                    flag = bool(flag)

                if flag:
                    continue

            polygon = np.asarray(polygon, dtype=np.float32)

            if polygon.size == 0:
                continue

            polygon = polygon.reshape(-1, 2)
            polygon = polygon[np.isfinite(polygon).all(axis=1)]

            if len(polygon) == 0:
                continue

            center = polygon.mean(axis=0)

            center_x = center[0] * scale_x
            center_y = center[1] * scale_y

            if (
                center_x >= 0.0
                and center_x < output_width
                and center_y >= 0.0
                and center_y < output_height
            ):
                centers.append(
                    np.array(
                        [center_x, center_y],
                        dtype=np.float32
                    )
                )

    return centers



def build_proposal_targets(
    output,
    batch,
    match_radius=8.0,
    return_matches=False
):
    proposal_output = output["proposal_output"]
    pred_centers = proposal_output["proposal_centers"]
    device = pred_centers.device
    batch_size, num_proposals = pred_centers.shape[:2]

    proposal_targets = torch.zeros(
        batch_size,
        num_proposals,
        device=device,
        dtype=torch.float32
    )

    proposal_center_targets = torch.zeros(
        batch_size,
        num_proposals,
        2,
        device=device,
        dtype=torch.float32
    )

    matched_gt_indices = torch.full(
        (batch_size, num_proposals),
        -1,
        device=device,
        dtype=torch.long
    )

    height, width = proposal_output["boundary_score"].shape[-2:]
    polygons_batch = batch["polygons"]
    ignore_batch = batch["ignore_flags"]

    for b in range(batch_size):
        sample_polygons = polygons_batch[b]
        sample_ignore = ignore_batch[b]

        if not sample_polygons or sample_polygons[0] is None:
            continue

        reference_polygons = sample_polygons[0]
        reference_ignore = (
            sample_ignore[0]
            if sample_ignore is not None and len(sample_ignore) > 0
            else None
        )

        valid_polygons = []
        gt_centers = []

        for polygon_index, polygon in enumerate(reference_polygons):
            ignored = False

            if (
                reference_ignore is not None
                and polygon_index < len(reference_ignore)
            ):
                flag = reference_ignore[polygon_index]

                if isinstance(flag, (list, tuple, np.ndarray)):
                    flag_array = np.asarray(flag).reshape(-1)
                    ignored = bool(flag_array[0]) if len(flag_array) else False
                else:
                    ignored = bool(flag)

            if ignored:
                continue

            polygon = np.asarray(polygon, dtype=np.float32)

            if polygon.size == 0:
                continue

            polygon = polygon.reshape(-1, 2)
            polygon = polygon[np.isfinite(polygon).all(axis=1)]

            if len(polygon) < 3:
                continue

            center = polygon.mean(axis=0)
            center_x = center[0] * width / 1280.0
            center_y = center[1] * height / 720.0

            if not (
                0.0 <= center_x < width
                and 0.0 <= center_y < height
            ):
                continue

            valid_polygons.append(polygon_index)
            gt_centers.append([center_x, center_y])

        if not gt_centers:
            continue

        gt_centers_tensor = torch.tensor(
            gt_centers,
            device=device,
            dtype=torch.float32
        )

        predicted_centers = pred_centers[b].float()

        distances = torch.cdist(
            predicted_centers,
            gt_centers_tensor,
            p=2
        )

        min_distance, nearest_valid_index = distances.min(dim=1)
        positive = min_distance <= match_radius

        proposal_targets[b] = positive.float()

        positive_indices = torch.nonzero(
            positive,
            as_tuple=True
        )[0]

        if positive_indices.numel() > 0:
            matched_centers = gt_centers_tensor[
                nearest_valid_index[positive_indices]
            ]

            proposal_center_targets[b, positive_indices] = matched_centers

            original_gt_indices = torch.tensor(
                valid_polygons,
                device=device,
                dtype=torch.long
            )

            matched_gt_indices[b, positive_indices] = (
                original_gt_indices[
                    nearest_valid_index[positive_indices]
                ]
            )

    if return_matches:
        return (
            proposal_targets,
            proposal_center_targets,
            matched_gt_indices
        )

    return proposal_targets, proposal_center_targets



import cv2
from configs.config import MAX_TEXT_LENGTH, CHAR2IDX, EOS_IDX, PAD_IDX

PROP_H = 64
PROP_W = 256
ORI_H = 720
ORI_W = 1280


def encode_transcript(text, max_length=MAX_TEXT_LENGTH):
    if text is None:
        text = ""

    encoded = []

    for c in text:
        if c in CHAR2IDX:
            encoded.append(CHAR2IDX[c])

    encoded = encoded[:max_length - 1]
    encoded.append(EOS_IDX)

    while len(encoded) < max_length:
        encoded.append(PAD_IDX)

    return torch.tensor(encoded, dtype=torch.long)


def polygon_center(poly):
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    cx = float(np.mean(poly[:, 0]))
    cy = float(np.mean(poly[:, 1]))
    return cx, cy


def resample_polygon_contour(polygon, max_vertices=32):
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    polygon = polygon[np.isfinite(polygon).all(axis=1)]

    if len(polygon) < 3:
        return None

    if np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]

    if len(polygon) < 3:
        return None

    keep = np.ones(len(polygon), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(polygon, axis=0), axis=1) > 1e-6
    polygon = polygon[keep]

    if len(polygon) < 3:
        return None

    next_points = np.roll(polygon, -1, axis=0)
    segment_lengths = np.linalg.norm(next_points - polygon, axis=1)
    perimeter = segment_lengths.sum()

    if not np.isfinite(perimeter) or perimeter <= 1e-6:
        return None

    if len(polygon) <= max_vertices:
        return polygon.astype(np.float32)

    cumulative = np.concatenate([
        np.array([0.0], dtype=np.float32),
        np.cumsum(segment_lengths, dtype=np.float32)
    ])

    distances = np.linspace(
        0.0,
        perimeter,
        num=max_vertices,
        endpoint=False,
        dtype=np.float32
    )

    result = []

    for distance in distances:
        segment_index = np.searchsorted(
            cumulative[1:],
            distance,
            side="right"
        )
        segment_index = min(segment_index, len(polygon) - 1)

        segment_length = segment_lengths[segment_index]

        if segment_length <= 1e-6:
            point = polygon[segment_index]
        else:
            ratio = (
                distance - cumulative[segment_index]
            ) / segment_length

            point = (
                polygon[segment_index]
                + ratio * (
                    next_points[segment_index]
                    - polygon[segment_index]
                )
            )

        result.append(point)

    return np.asarray(result, dtype=np.float32)


def build_polygon_targets(
    polygons,
    ignore_flags=None,
    output_height=64,
    output_width=256,
    image_height=720.0,
    image_width=1280.0,
    max_vertices=32
):
    if (
        polygons is None
        or len(polygons) == 0
        or polygons[0] is None
    ):
        reference_polygons = []
    else:
        reference_polygons = polygons[0]

    if (
        ignore_flags is not None
        and len(ignore_flags) > 0
        and ignore_flags[0] is not None
    ):
        reference_ignore = ignore_flags[0]
    else:
        reference_ignore = []

    vertices = []
    vertex_masks = []
    stop_targets = []
    lengths = []
    original_indices = []

    scale = np.array(
        [
            output_width / image_width,
            output_height / image_height
        ],
        dtype=np.float32
    )

    for original_index, raw_polygon in enumerate(reference_polygons):
        if original_index < len(reference_ignore):
            flag = reference_ignore[original_index]

            if isinstance(flag, (list, tuple, np.ndarray)):
                flag = np.asarray(flag).reshape(-1)
                ignored = bool(flag[0]) if len(flag) else False
            else:
                ignored = bool(flag)

            if ignored:
                continue

        contour = resample_polygon_contour(
            raw_polygon,
            max_vertices=max_vertices
        )

        if contour is None:
            continue

        contour = contour * scale
        contour[:, 0] = np.clip(contour[:, 0], 0, output_width - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, output_height - 1)

        length = len(contour)

        padded_vertices = np.zeros(
            (max_vertices, 2),
            dtype=np.float32
        )
        vertex_mask = np.zeros(max_vertices, dtype=np.float32)
        stop_target = np.zeros(max_vertices, dtype=np.float32)

        padded_vertices[:length] = contour
        vertex_mask[:length] = 1.0
        stop_target[length - 1] = 1.0

        vertices.append(padded_vertices)
        vertex_masks.append(vertex_mask)
        stop_targets.append(stop_target)
        lengths.append(length)
        original_indices.append(original_index)

    if not vertices:
        return {
            "vertices": np.zeros(
                (0, max_vertices, 2), dtype=np.float32
            ),
            "vertex_mask": np.zeros(
                (0, max_vertices), dtype=np.float32
            ),
            "stop_targets": np.zeros(
                (0, max_vertices), dtype=np.float32
            ),
            "lengths": np.zeros((0,), dtype=np.int64),
            "original_indices": np.zeros((0,), dtype=np.int64)
        }

    return {
        "vertices": np.stack(vertices).astype(np.float32),
        "vertex_mask": np.stack(vertex_masks).astype(np.float32),
        "stop_targets": np.stack(stop_targets).astype(np.float32),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "original_indices": np.asarray(
            original_indices, dtype=np.int64
        )
    }



def resample_polygon_contour(polygon, max_vertices=32):
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    polygon = polygon[np.isfinite(polygon).all(axis=1)]

    if len(polygon) > 1:
        keep = np.ones(len(polygon), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(polygon, axis=0), axis=1) > 1e-6
        polygon = polygon[keep]

    if len(polygon) < 3:
        return None

    if len(polygon) <= max_vertices:
        return polygon

    closed = np.concatenate([polygon, polygon[:1]], axis=0)
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = float(cumulative[-1])

    if perimeter <= 1e-6:
        return None

    distances = np.linspace(0.0, perimeter, max_vertices, endpoint=False)
    resampled = []

    for distance in distances:
        segment_index = np.searchsorted(cumulative, distance, side="right") - 1
        segment_index = min(segment_index, len(polygon) - 1)

        segment_start = closed[segment_index]
        segment_end = closed[segment_index + 1]
        segment_length = segment_lengths[segment_index]

        if segment_length <= 1e-6:
            point = segment_start
        else:
            ratio = (distance - cumulative[segment_index]) / segment_length
            point = segment_start + ratio * (segment_end - segment_start)

        resampled.append(point)

    return np.asarray(resampled, dtype=np.float32)


def build_polygon_targets(
    polygons,
    ignore_flags=None,
    output_height=64,
    output_width=256,
    image_height=720.0,
    image_width=1280.0,
    max_vertices=32
):
    if max_vertices < 3:
        raise ValueError("max_vertices must be at least 3")

    if polygons is None or len(polygons) == 0 or polygons[0] is None:
        return {
            "vertices": np.zeros((0, max_vertices, 2), dtype=np.float32),
            "vertex_mask": np.zeros((0, max_vertices), dtype=np.float32),
            "stop_targets": np.zeros((0, max_vertices), dtype=np.float32),
            "polygon_mask": np.zeros((0,), dtype=np.float32),
            "lengths": np.zeros((0,), dtype=np.int64)
        }

    reference_polygons = polygons[0]

    if ignore_flags is not None and len(ignore_flags) > 0:
        reference_ignore = ignore_flags[0]
    else:
        reference_ignore = None

    scale = np.asarray(
        [output_width / image_width, output_height / image_height],
        dtype=np.float32
    )

    vertex_targets = []
    vertex_masks = []
    stop_targets = []
    lengths = []

    for index, raw_polygon in enumerate(reference_polygons):
        if reference_ignore is not None and index < len(reference_ignore):
            flag = reference_ignore[index]

            if isinstance(flag, (list, tuple, np.ndarray)):
                flag_values = np.asarray(flag).reshape(-1)
                flag = bool(flag_values[0]) if len(flag_values) else False

            if bool(flag):
                continue

        polygon = resample_polygon_contour(raw_polygon, max_vertices)

        if polygon is None:
            continue

        polygon = polygon * scale
        polygon[:, 0] = np.clip(polygon[:, 0], 0, output_width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, output_height - 1)

        length = len(polygon)

        vertices = np.zeros((max_vertices, 2), dtype=np.float32)
        vertex_mask = np.zeros((max_vertices,), dtype=np.float32)
        stop = np.zeros((max_vertices,), dtype=np.float32)

        vertices[:length] = polygon
        vertex_mask[:length] = 1.0
        stop[length - 1] = 1.0

        vertex_targets.append(vertices)
        vertex_masks.append(vertex_mask)
        stop_targets.append(stop)
        lengths.append(length)

    if not vertex_targets:
        return {
            "vertices": np.zeros((0, max_vertices, 2), dtype=np.float32),
            "vertex_mask": np.zeros((0, max_vertices), dtype=np.float32),
            "stop_targets": np.zeros((0, max_vertices), dtype=np.float32),
            "polygon_mask": np.zeros((0,), dtype=np.float32),
            "lengths": np.zeros((0,), dtype=np.int64)
        }

    return {
        "vertices": np.stack(vertex_targets).astype(np.float32),
        "vertex_mask": np.stack(vertex_masks).astype(np.float32),
        "stop_targets": np.stack(stop_targets).astype(np.float32),
        "polygon_mask": np.ones((len(lengths),), dtype=np.float32),
        "lengths": np.asarray(lengths, dtype=np.int64)
    }



def draw_gaussian(heatmap, center_x, center_y, sigma=2.0):
    H, W = heatmap.shape
    radius = int(max(1, 3 * sigma))

    x0 = max(0, int(center_x) - radius)
    x1 = min(W - 1, int(center_x) + radius)
    y0 = max(0, int(center_y) - radius)
    y1 = min(H - 1, int(center_y) + radius)

    if x0 > x1 or y0 > y1:
        return

    xs = np.arange(x0, x1 + 1)
    ys = np.arange(y0, y1 + 1)
    xx, yy = np.meshgrid(xs, ys)

    gaussian = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2)
        / (2 * sigma * sigma)
    )

    heatmap[y0:y1 + 1, x0:x1 + 1] = np.maximum(
        heatmap[y0:y1 + 1, x0:x1 + 1],
        gaussian
    )




def build_detection_targets(polygons, ignore_flags=None):
    boundary = np.zeros((PROP_H, PROP_W), dtype=np.float32)
    center = np.zeros((PROP_H, PROP_W), dtype=np.float32)
    polygon_count = 0

    if polygons is None or len(polygons) == 0 or polygons[0] is None:
        return boundary, center, polygon_count

    reference_polygons = polygons[0]

    if ignore_flags is not None and len(ignore_flags) > 0:
        reference_ignore = ignore_flags[0]
    else:
        reference_ignore = None

    scale_x = PROP_W / float(ORI_W)
    scale_y = PROP_H / float(ORI_H)

    for index, raw_polygon in enumerate(reference_polygons):
        if reference_ignore is not None and index < len(reference_ignore):
            flag = reference_ignore[index]

            if isinstance(flag, (list, tuple, np.ndarray)):
                flag_values = np.asarray(flag).reshape(-1)
                flag = bool(flag_values[0]) if len(flag_values) else False

            if bool(flag):
                continue

        polygon = resample_polygon_contour(raw_polygon, 32)

        if polygon is None:
            continue

        polygon_map = polygon.copy()
        polygon_map[:, 0] *= scale_x
        polygon_map[:, 1] *= scale_y
        polygon_map[:, 0] = np.clip(polygon_map[:, 0], 0, PROP_W - 1)
        polygon_map[:, 1] = np.clip(polygon_map[:, 1], 0, PROP_H - 1)

        polygon_int = np.round(polygon_map).astype(np.int32)

        cv2.polylines(
            boundary,
            [polygon_int],
            isClosed=True,
            color=1.0,
            thickness=2
        )

        cx, cy = polygon_center(polygon_map)
        draw_gaussian(center, cx, cy, sigma=2.0)
        polygon_count += 1

    boundary = np.clip(boundary, 0.0, 1.0)
    center = np.clip(center, 0.0, 1.0)

    return boundary, center, polygon_count

