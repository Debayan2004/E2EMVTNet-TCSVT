
import cv2
import numpy as np

from configs.config import (
    DET_INPUT_HEIGHT,
    DET_INPUT_WIDTH,
    REC_INPUT_HEIGHT,
    REC_INPUT_WIDTH,
)


def resize_detection(image):
    return cv2.resize(
        image,
        (DET_INPUT_WIDTH, DET_INPUT_HEIGHT)
    )


def resize_word(image):
    h, w = image.shape[:2]

    scale = min(
        REC_INPUT_WIDTH / w,
        REC_INPUT_HEIGHT / h
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    image = cv2.resize(
        image,
        (new_w, new_h)
    )

    canvas = np.ones(
        (
            REC_INPUT_HEIGHT,
            REC_INPUT_WIDTH,
            3
        ),
        dtype=np.uint8
    ) * 255

    x = (REC_INPUT_WIDTH - new_w) // 2
    y = (REC_INPUT_HEIGHT - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = image

    return canvas


def order_points(points):
    points = np.array(
        points,
        dtype=np.float32
    ).reshape(4, 2)

    rect = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    differences = np.diff(points, axis=1)
    rect[1] = points[np.argmin(differences)]
    rect[3] = points[np.argmax(differences)]

    return rect


def rectify_polygon(image, polygon):
    points = order_points(polygon)

    top_left, top_right, bottom_right, bottom_left = points

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)

    width = max(int(max(width_a, width_b)), 2)
    height = max(int(max(height_a, height_b)), 2)

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(
        points,
        destination
    )

    crop = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return crop
