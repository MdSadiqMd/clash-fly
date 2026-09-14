"""COCO -> YOLO conversion with class aliasing, on a two-image synthetic set."""

import json

import numpy as np
import cv2

from flyclash.game import dataset


def test_coco_to_yolo_maps_and_drops_classes(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(imgs / name), np.zeros((100, 200, 3), np.uint8))
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 200, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 200, "height": 100},
        ],
        "categories": [
            {"id": 1, "name": "Canon"},
            {"id": 2, "name": "TH13"},
            {"id": 3, "name": "Unknown"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [50, 25, 20, 10]},
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 5, 5]},
            {"image_id": 2, "category_id": 2, "bbox": [100, 50, 40, 20]},
        ],
    }
    (imgs / "ann.json").write_text(json.dumps(coco))
    root = tmp_path / "yolo"
    assert dataset.coco_to_yolo(imgs / "ann.json", imgs, "train", root) == 2
    a = (root / "labels" / "train" / "a.txt").read_text().splitlines()
    assert len(a) == 1 and a[0].startswith(
        f"{dataset.CLASSES.index('Cannon')} 0.300000 0.300000 0.100000 0.100000"
    )
    b = (root / "labels" / "train" / "b.txt").read_text()
    assert b.startswith(f"{dataset.CLASSES.index('TownHall')} ")
    assert "TownHall" in dataset.write_data_yaml(root).read_text()
