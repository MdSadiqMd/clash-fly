"""Fetch public Clash of Clans detection data and convert it to YOLO format.

Public sets are high-Town-Hall (find-this-base, 125 images, CC-BY 4.0, mirrored on
Hugging Face as keremberke/clash-of-clans-object-detection). Their class names are
mapped onto ours; classes we do not model are dropped. Add your own Roboflow YOLO
export (any TH level) with `add_yolo_export` and everything merges into one data.yaml.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from flyclash.config import DATA
from flyclash.game.base_reader import CANONICAL_CLASSES, canonical_class

HF_BASE = "https://huggingface.co/datasets/keremberke/clash-of-clans-object-detection/resolve/main/data/"
HF_SPLITS = {"train": "train.zip", "val": "valid.zip", "test": "test.zip"}
UA = "OpenAI File Downloader, XaiImageApiFetch/1.0"
CLASSES = CANONICAL_CLASSES
YOLO_ROOT = DATA / "yolo" / "coc"
canonical = canonical_class


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    return dest


def coco_to_yolo(
    coco_json: Path,
    images_dir: Path,
    split: str,
    root: Path = YOLO_ROOT,
    classes: list[str] = CLASSES,
    canonical=canonical_class,
) -> int:
    """Write YOLO txt labels + copy images for one split. Returns number of images written."""
    CLASSES, canonical = classes, canonical  # noqa: N806 - shadow module defaults with the caller's game
    coco = json.loads(coco_json.read_text())
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    img_out, lbl_out = root / "images" / split, root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in coco["images"]:
        src = images_dir / im["file_name"]
        if not src.exists():
            continue
        lines = []
        for a in by_image.get(im["id"], []):
            cls = canonical(cat_name[a["category_id"]])
            if cls is None:
                continue
            x, y, w, h = a["bbox"]
            cx, cy = (x + w / 2) / im["width"], (y + h / 2) / im["height"]
            lines.append(
                f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {w / im['width']:.6f} {h / im['height']:.6f}"
            )
        shutil.copy(src, img_out / src.name)
        (lbl_out / (src.stem + ".txt")).write_text("\n".join(lines))
        n += 1
    return n


def write_data_yaml(root: Path = YOLO_ROOT, classes: list[str] = CLASSES) -> Path:
    p = root / "data.yaml"
    p.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes))
    )
    return p


def fetch_huggingface(root: Path = YOLO_ROOT) -> dict[str, int]:
    counts = {}
    for split, zname in HF_SPLITS.items():
        z = _download(HF_BASE + zname, DATA / "raw" / "coc_hf" / zname)
        ex = z.with_suffix("")
        if not ex.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(ex)
        jsons = list(ex.rglob("*.json"))
        if not jsons:
            continue
        counts[split] = coco_to_yolo(jsons[0], jsons[0].parent, split, root)
    write_data_yaml(root)
    return counts


def add_yolo_export(
    export_dir: Path,
    root: Path = YOLO_ROOT,
    classes: list[str] = CLASSES,
    canonical=canonical_class,
) -> dict[str, int]:
    """Merge a Roboflow 'YOLOv8' export (data.yaml + train/valid/test folders) into our set,
    remapping its class ids by name. Works for any game: pass its class list and alias function."""
    import yaml  # ships with ultralytics

    CLASSES = classes  # noqa: N806
    names = yaml.safe_load((export_dir / "data.yaml").read_text())["names"]
    names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    counts = {}
    for src_split, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        lbl_dir = export_dir / src_split / "labels"
        if not lbl_dir.exists():
            continue
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for lbl in lbl_dir.glob("*.txt"):
            lines = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = canonical(names[int(parts[0])])
                if cls is not None:
                    lines.append(" ".join([str(CLASSES.index(cls)), *parts[1:5]]))
            img = next(
                (p for p in (export_dir / src_split / "images").glob(lbl.stem + ".*")),
                None,
            )
            if img is None:
                continue
            shutil.copy(img, root / "images" / split / img.name)
            (root / "labels" / split / lbl.name).write_text("\n".join(lines))
            n += 1
        counts[split] = n
    write_data_yaml(root, classes)
    return counts


def train(
    data_yaml: Path,
    out: Path,
    epochs: int = 50,
    imgsz: int = 960,
    model: str = "yolov8n.pt",
    run_name: str = "coc",
) -> Path:
    import torch
    from ultralytics import YOLO

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("0" if torch.cuda.is_available() else "cpu")
    )
    m = YOLO(model)
    m.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(DATA / "runs"),
        name=run_name,
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    best = DATA / "runs" / run_name / "weights" / "best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    return out
