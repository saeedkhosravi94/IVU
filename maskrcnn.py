#!/usr/bin/env python3
"""
Train + evaluate a TorchVision Mask R-CNN on the TACO dataset (COCO format).

Assumptions:
- Images live under:   TACO-master/data/<file_name from COCO JSON>
- COCO JSON is:        TACO-master/data/annotations.json (default)

Notes:
- TACO images may contain EXIF orientation tags. We intentionally DO NOT apply
  `ImageOps.exif_transpose` here because the COCO coordinates in the JSON align
  with the raw pixel coordinate system as stored on disk.
- For full COCO mAP/mAR evaluation, install `pycocotools`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torchvision.transforms import v2 as T


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _poly_to_mask(polys: List[List[float]], h: int, w: int) -> torch.Tensor:
    """
    Rasterize polygons into a binary mask using PIL (no extra deps).
    polys: list of flattened [x1,y1,x2,y2,...] polygon lists.
    Returns: (H,W) uint8 tensor {0,1}.
    """
    from PIL import ImageDraw

    m = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(m)
    for p in polys:
        if len(p) < 6:
            continue
        xy = [(p[i], p[i + 1]) for i in range(0, len(p), 2)]
        draw.polygon(xy, outline=1, fill=1)
    return torch.from_numpy(np.array(m, dtype=np.uint8))


@dataclass(frozen=True)
class TacoIndex:
    img_by_id: Dict[int, Dict[str, Any]]
    anns_by_img: Dict[int, List[Dict[str, Any]]]
    cat_ids_sorted: List[int]
    cat_id_to_label: Dict[int, int]  # COCO category_id -> contiguous label in [1..K]
    label_to_cat_id: Dict[int, int]


def build_index(coco: Dict[str, Any]) -> TacoIndex:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    img_by_id: Dict[int, Dict[str, Any]] = {}
    for im in images:
        if isinstance(im, dict) and "id" in im:
            img_by_id[int(im["id"])] = im

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {i: [] for i in img_by_id.keys()}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        if "image_id" not in ann:
            continue
        img_id = int(ann["image_id"])
        if img_id in anns_by_img:
            anns_by_img[img_id].append(ann)

    cat_ids = []
    for c in categories:
        if isinstance(c, dict) and "id" in c:
            cat_ids.append(int(c["id"]))
    cat_ids_sorted = sorted(set(cat_ids))
    cat_id_to_label = {cid: i + 1 for i, cid in enumerate(cat_ids_sorted)}  # 0 reserved for background
    label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}
    return TacoIndex(
        img_by_id=img_by_id,
        anns_by_img=anns_by_img,
        cat_ids_sorted=cat_ids_sorted,
        cat_id_to_label=cat_id_to_label,
        label_to_cat_id=label_to_cat_id,
    )


class TacoInstanceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        ann_path: Path,
        image_ids: List[int],
        transforms: Optional[Any] = None,
        train: bool = False,
    ):
        self.dataset_dir = dataset_dir
        self.ann_path = ann_path
        self.coco = _load_json(ann_path)
        self.index = build_index(self.coco)
        self.image_ids = image_ids
        self.transforms = transforms
        self.train = train

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = int(self.image_ids[idx])
        im = self.index.img_by_id[img_id]
        file_name = str(im["file_name"])
        img_path = self.dataset_dir / file_name
        img = Image.open(img_path).convert("RGB")  # keep raw orientation

        w = int(im.get("width", img.size[0]))
        h = int(im.get("height", img.size[1]))

        anns = self.index.anns_by_img.get(img_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[torch.Tensor] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            cat_id = int(ann.get("category_id", -1))
            if cat_id not in self.index.cat_id_to_label:
                continue
            bbox = ann.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x, y, bw, bh = map(float, bbox)
            if not all(math.isfinite(v) for v in (x, y, bw, bh)):
                continue
            if bw <= 1 or bh <= 1:
                continue

            # Clamp slightly-negative coordinates (some TACO bboxes have -1.0).
            x0 = max(0.0, x)
            y0 = max(0.0, y)
            x1 = min(float(w), x0 + max(0.0, bw))
            y1 = min(float(h), y0 + max(0.0, bh))
            if x1 <= x0 or y1 <= y0:
                continue

            boxes.append([x0, y0, x1, y1])
            labels.append(self.index.cat_id_to_label[cat_id])
            areas.append(float(ann.get("area", (x1 - x0) * (y1 - y0))))
            iscrowd.append(int(ann.get("iscrowd", 0)))

            seg = ann.get("segmentation")
            if isinstance(seg, list):
                # list of polygons or flattened polygon
                if len(seg) > 0 and all(isinstance(v, (int, float)) for v in seg):
                    polys = [list(map(float, seg))]
                else:
                    polys = [list(map(float, p)) for p in seg if isinstance(p, list)]
                masks.append(_poly_to_mask(polys, h=h, w=w))
            else:
                # If segmentation is missing or RLE, create empty mask.
                masks.append(torch.zeros((h, w), dtype=torch.uint8))

        target: Dict[str, Any] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "masks": torch.stack(masks, dim=0) if len(masks) else torch.zeros((0, h, w), dtype=torch.uint8),
        }

        img_t = T.ToImage()(img)  # uint8 tensor (C,H,W)
        img_t = T.ToDtype(torch.float32, scale=True)(img_t)  # float32 in [0,1]

        # Keep augmentation minimal and stable to avoid NaNs.
        # TorchVision detection models already apply resizing internally.
        if self.train and torch.rand(()) < 0.5:
            # horizontal flip
            img_t = torch.flip(img_t, dims=[2])
            if target["boxes"].numel() > 0:
                boxes_t = target["boxes"].clone()
                # boxes are xyxy
                boxes_t[:, [0, 2]] = float(w) - boxes_t[:, [2, 0]]
                target["boxes"] = boxes_t
            if target["masks"].numel() > 0:
                target["masks"] = torch.flip(target["masks"], dims=[2])

        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)

        return img_t, target


def build_transforms(train: bool, min_size: int = 640, max_size: int = 1024):
    # TorchVision v2 transforms that keep boxes/masks consistent.
    if train:
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ScaleJitter(target_size=(min_size, min_size), scale_range=(0.6, 1.2)),
                T.ClampBoundingBoxes(),
            ]
        )
    return T.Compose(
        [
            T.Resize(size=min_size, max_size=max_size),
            T.ClampBoundingBoxes(),
        ]
    )


def get_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden, num_classes
    )
    return model


@torch.no_grad()
def evaluate_simple(model, data_loader, device) -> Dict[str, float]:
    """
    Lightweight evaluation: average losses on the eval set (not COCO mAP).
    Useful to verify the pipeline is working.
    """
    # In TorchVision, losses are returned only in train mode.
    # We temporarily switch to train() under no_grad() to compute a validation loss.
    was_training = model.training
    model.train()
    losses = []
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        if not isinstance(loss_dict, dict):
            # If something unexpected happens, skip this batch.
            continue
        loss = float(sum(loss_dict.values()).item())
        losses.append(loss)
    if not was_training:
        model.eval()
    return {"loss": float(np.mean(losses)) if losses else float("nan")}


def try_coco_eval(model, dataset: TacoInstanceDataset, data_loader, device) -> Optional[Dict[str, float]]:
    """
    COCO mAP/mAR via pycocotools if available. Returns dict or None if dependency missing.
    """
    try:
        from pycocotools.cocoeval import COCOeval  # type: ignore
        from pycocotools.coco import COCO  # type: ignore
        import pycocotools.mask as mask_utils  # type: ignore
    except Exception:
        return None

    # Build COCO object from the same JSON
    coco_gt = COCO(str(dataset.ann_path))

    model.eval()
    results = []
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            masks = out.get("masks")
            if masks is not None:
                masks = masks.detach().cpu().numpy()  # [N,1,H,W]
            for i in range(boxes.shape[0]):
                x0, y0, x1, y1 = boxes[i].tolist()
                w = max(0.0, x1 - x0)
                h = max(0.0, y1 - y0)
                label = int(labels[i])
                cat_id = dataset.index.label_to_cat_id.get(label, None)
                if cat_id is None:
                    continue
                r = {
                    "image_id": img_id,
                    "category_id": int(cat_id),
                    "bbox": [x0, y0, w, h],
                    "score": float(scores[i]),
                }
                if masks is not None:
                    m = masks[i, 0] > 0.5
                    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("ascii")
                    r["segmentation"] = rle
                results.append(r)

    if len(results) == 0:
        return {"coco_eval": float("nan")}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    return {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
    }


def train_one_epoch(model, optimizer, data_loader, device, epoch: int, print_freq: int = 50) -> float:
    model.train()
    losses = []
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            # Skip bad batches instead of poisoning the run.
            print(f"epoch={epoch} iter={i+1}: non-finite loss ({loss.item()}), skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (i + 1) % print_freq == 0:
            print(f"epoch={epoch} iter={i+1}/{len(data_loader)} loss={np.mean(losses[-print_freq:]):.4f}")
    return float(np.mean(losses)) if losses else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/eval TorchVision Mask R-CNN on TACO.")
    parser.add_argument("--dataset_dir", type=Path, default=Path("TACO-master/data"))
    parser.add_argument("--annotations", type=Path, default=Path("TACO-master/data/annotations.json"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained COCO weights")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/maskrcnn_taco"))
    parser.add_argument("--limit_images", type=int, default=0, help="If >0, limit to this many images total (debug)")
    parser.add_argument("--coco_eval", action="store_true", help="Compute COCO mAP (requires pycocotools)")
    args = parser.parse_args()

    seed_everything(args.seed)

    coco = _load_json(args.annotations)
    index = build_index(coco)
    image_ids = sorted(index.img_by_id.keys())
    if args.limit_images and args.limit_images > 0:
        image_ids = image_ids[: args.limit_images]

    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    n_train = int(len(image_ids) * args.train_frac)
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:]

    num_classes = len(index.cat_ids_sorted) + 1  # + background
    print(f"Images: {len(image_ids)} (train={len(train_ids)} val={len(val_ids)})  classes={num_classes-1}+bg")

    train_ds = TacoInstanceDataset(
        dataset_dir=args.dataset_dir,
        ann_path=args.annotations,
        image_ids=train_ids,
        transforms=None,
        train=True,
    )
    val_ds = TacoInstanceDataset(
        dataset_dir=args.dataset_dir,
        ann_path=args.annotations,
        image_ids=val_ids,
        transforms=None,
        train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = get_model(num_classes=num_classes, pretrained=args.pretrained).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch=epoch, print_freq=20)
        lr_scheduler.step()
        val_metrics = evaluate_simple(model, val_loader, device)
        print(f"epoch={epoch} train_loss={avg_loss:.4f} val_loss={val_metrics['loss']:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )

    if args.coco_eval:
        coco_metrics = try_coco_eval(model, val_ds, val_loader, device)
        if coco_metrics is None:
            print("COCO eval requested but pycocotools is not installed. Install with: pip install pycocotools")
        else:
            print("COCO metrics:", coco_metrics)

    print(f"Saved checkpoint to: {ckpt_path}")
    return 0


if __name__ == "__main__":
    # Avoid MKL thread explosions on some macOS setups.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())

