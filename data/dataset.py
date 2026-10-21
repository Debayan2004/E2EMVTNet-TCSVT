import os
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import (
    NUM_VIEWS,
    MAX_TEXT_LENGTH,
    CHAR2IDX,
    EOS_IDX,
    PAD_IDX,
)


class SpottingDataset(Dataset):
    def __init__(self, scene_dirs, transform=None, max_text_length=MAX_TEXT_LENGTH):
        super().__init__()

        self.scene_dirs = sorted(scene_dirs)
        self.transform = transform
        self.max_text_length = max_text_length
        self.samples = []
        self.image_cache = {}

        self.build_dataset()

        print("=" * 80)
        print("E2EMVTNet Spotting Dataset")
        print("=" * 80)
        print("Scenes :", len(self.scene_dirs))
        print("Samples:", len(self.samples))
        print("=" * 80)

    def encode_text(self, text):
        encoded = [
            CHAR2IDX[c]
            for c in text
            if c in CHAR2IDX
        ]

        encoded = encoded[:self.max_text_length - 1]
        encoded.append(EOS_IDX)
        encoded += [PAD_IDX] * (
            self.max_text_length - len(encoded)
        )

        return torch.LongTensor(encoded)

    def load_image(self, path):
        if path not in self.image_cache:
            img = cv2.imread(path)

            if img is None:
                raise FileNotFoundError(path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image_cache[path] = img

        return self.image_cache[path]

    def build_dataset(self):
        self.samples = []

        for scene_dir in self.scene_dirs:
            json_files = list(Path(scene_dir).glob("*.json"))

            if not json_files:
                raise FileNotFoundError(
                    f"No JSON annotation file found in {scene_dir}"
                )

            with open(json_files[0], "r", encoding="utf-8") as f:
                coco = json.load(f)

            image_paths = {
                img["id"]: os.path.join(scene_dir, img["file_name"])
                for img in coco["images"]
            }

            anns_per_view = defaultdict(list)

            for ann in coco["annotations"]:
                anns_per_view[ann["image_id"]].append(ann)

            for view in anns_per_view:
                anns_per_view[view] = sorted(
                    anns_per_view[view],
                    key=lambda x: x["id"]
                )

            view_maps = {}

            for view in range(1, NUM_VIEWS + 1):
                counter = defaultdict(int)
                mapping = {}

                for ann in anns_per_view[view]:
                    if ann["ignore"] == 1:
                        continue

                    counter[ann["text"]] += 1
                    occ = counter[ann["text"]]
                    mapping[(ann["text"], occ)] = ann

                view_maps[view] = mapping

            all_keys = set()

            for view in range(1, NUM_VIEWS + 1):
                all_keys.update(view_maps[view].keys())

            detection_images = []
            detection_polygons = []
            detection_bboxes = []
            detection_ignore = []

            for view in range(1, NUM_VIEWS + 1):
                detection_images.append(image_paths[view])

                polygons = []
                boxes = []
                ignores = []

                for ann in anns_per_view[view]:
                    polygon = np.asarray(
                        ann["segmentation"][0],
                        dtype=np.float32
                    ).reshape(-1, 2)

                    polygons.append(polygon)
                    boxes.append(ann["bbox"])
                    ignores.append(ann["ignore"])

                detection_polygons.append(polygons)
                detection_bboxes.append(boxes)
                detection_ignore.append(ignores)

            recognition_samples = []

            for key in sorted(all_keys):
                transcript = key[0]
                sample_views = []
                view_mask = []

                for view in range(1, NUM_VIEWS + 1):
                    if key in view_maps[view]:
                        ann = view_maps[view][key]

                        polygon = np.asarray(
                            ann["segmentation"][0],
                            dtype=np.float32
                        ).reshape(-1, 2)

                        sample_views.append({
                            "image_path": image_paths[view],
                            "polygon": polygon,
                            "bbox": ann["bbox"]
                        })

                        view_mask.append(1)
                    else:
                        sample_views.append(None)
                        view_mask.append(0)

                recognition_samples.append({
                    "text": transcript,
                    "views": sample_views,
                    "view_mask": view_mask
                })

            self.samples.append({
                "scene_dir": scene_dir,
                "detection_images": detection_images,
                "polygons": detection_polygons,
                "bboxes": detection_bboxes,
                "ignore_flags": detection_ignore,
                "recognition_samples": recognition_samples
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        detection_images = []

        for image_path in sample["detection_images"]:
            image = self.load_image(image_path)
            detection_images.append(image)

        return {
            "images": detection_images,
            "polygons": sample["polygons"],
            "bboxes": sample["bboxes"],
            "ignore_flags": sample["ignore_flags"],
            "recognition_samples": sample["recognition_samples"]
        }
    



def select_recognition_sample(recognition_samples, random_select=True):
    valid = []

    for item in recognition_samples:
        text = item.get("text", "")

        if text is None:
            continue

        text = str(text)

        if len(text.strip()) == 0:
            continue

        valid.append(text)

    if len(valid) == 0:
        return ""

    if random_select:
        idx = np.random.randint(0, len(valid))
        return valid[idx]

    return valid[0]


import cv2
import torch
import numpy as np

from configs.config import (
    DET_INPUT_HEIGHT,
    DET_INPUT_WIDTH,
    MAX_TEXT_LENGTH,
    PAD_IDX,
)
from data.targets import build_detection_targets, encode_transcript


def spotting_collate_fn(batch):
    images = []
    input_tokens = []
    target_tokens = []
    boundary_targets = []
    center_targets = []
    polygon_counts = []

    for sample in batch:
        scene_images = []

        for img in sample["images"]:
            img = cv2.resize(
                img,
                (DET_INPUT_WIDTH, DET_INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            img = img.astype(np.float32) / 255.0
            img = (img - np.array(
                [0.485, 0.456, 0.406], dtype=np.float32
            )) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32
            )

            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            scene_images.append(img)

        images.append(torch.stack(scene_images))

        boundary, center, count = build_detection_targets(
            sample["polygons"],
            sample["ignore_flags"]
        )

        boundary_targets.append(torch.from_numpy(boundary))
        center_targets.append(torch.from_numpy(center))
        polygon_counts.append(count)

        text = select_recognition_sample(
            sample["recognition_samples"],
            random_select=True,
        )

        target = encode_transcript(text, MAX_TEXT_LENGTH)
        decoder_input = torch.empty_like(target)
        decoder_input[0] = PAD_IDX
        decoder_input[1:] = target[:-1]

        input_tokens.append(decoder_input)
        target_tokens.append(target)

    images = torch.stack(images)
    input_tokens = torch.stack(input_tokens)
    target_tokens = torch.stack(target_tokens)
    boundary_targets = torch.stack(boundary_targets).unsqueeze(1)
    center_targets = torch.stack(center_targets).unsqueeze(1)

    return {
        "images": images,
        "input_tokens": input_tokens,
        "transcript": target_tokens,
        "target_tokens": target_tokens,
        "tokens": target_tokens,
        "boundary_heatmap": boundary_targets,
        "center_heatmap": center_targets,
        "polygon_counts": torch.tensor(polygon_counts, dtype=torch.float32),
        "polygons": [sample["polygons"] for sample in batch],
        "bboxes": [sample["bboxes"] for sample in batch],
        "ignore_flags": [sample["ignore_flags"] for sample in batch],
        "recognition_samples": [
            sample["recognition_samples"] for sample in batch
        ],
    }



