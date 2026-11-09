"""Adapt original COCO bitmaps to native YOLO losses without polygon conversion.

Images are RGB float32 CHW tensors in [0, 1]. Boxes are original-image absolute
xyxy, labels are sparse COCO category IDs, and each bitmap retains its annotation
identity, including disconnected components and overlaps with other instances.
This module never invokes an Ultralytics trainer, downloads assets, or writes data.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)
MASK_RATIO = 4
MASK_SAMPLING_POLICY = (
    "Per-instance uint8 bitmap; OpenCV INTER_NEAREST to the rounded letterbox content "
    "size; zero padding; OpenCV INTER_NEAREST from the square canvas to image_size/4. "
    "No polygon conversion, component splitting, instance merging, or foreground repair."
)


def _model_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    result = dict(names) if isinstance(names, Mapping) else dict(enumerate(names))
    if (
        set(result) != set(range(80))
        or any(type(key) is not int for key in result)
        or any(not isinstance(value, str) or not value for value in result.values())
        or len(set(result.values())) != 80
    ):
        raise ValueError("The unchanged COCO head requires 80 unique names indexed 0..79.")
    return result


def _class_mapping(
    names: Mapping[int, str] | Sequence[str],
    categories: Mapping[int, Any] | Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    by_name = {name: index for index, name in _model_names(names).items()}
    rows = (
        categories.items()
        if isinstance(categories, Mapping)
        else ((row["id"], row) for row in categories)
    )
    mapping = {}
    for key, row in rows:
        name = row.get("name") if isinstance(row, Mapping) else getattr(row, "name", None)
        if type(key) is not int or key < 0 or key in mapping or name not in by_name:
            raise ValueError("COCO categories need unique sparse IDs and matching model names.")
        mapping[key] = by_name[name]
    return mapping


def _target_tensors(
    target: Mapping[str, Any], height: int, width: int, mapping: dict[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    boxes = torch.as_tensor(target["boxes"]).detach().cpu().to(torch.float32).clone()
    labels = torch.as_tensor(target["labels"]).detach().cpu()
    masks = torch.as_tensor(target["masks"]).detach().cpu()
    ids = (
        torch.as_tensor(
            target["annotation_id"] if "annotation_id" in target else target["annotation_ids"]
        )
        .detach()
        .cpu()
    )
    if boxes.numel() == 0:
        boxes = boxes.reshape(0, 4)
    count = len(boxes)
    if (
        boxes.shape != (count, 4)
        or labels.shape != (count,)
        or masks.shape != (count, height, width)
        or ids.shape != (count,)
        or labels.dtype not in (torch.int32, torch.int64)
        or ids.dtype not in (torch.int32, torch.int64)
        or masks.dtype not in (torch.uint8, torch.bool)
        or not torch.isfinite(boxes).all()
    ):
        raise ValueError("Boxes, sparse labels, bitmap masks and annotation IDs must align.")
    if (ids < 0).any() or len(ids.unique()) != count:
        raise ValueError("Annotation IDs must be nonnegative and unique within an image.")
    if not ((masks == 0) | (masks == 1)).all():
        raise ValueError("Original COCO masks must contain binary 0/1 values.")
    if count and not masks.reshape(count, -1).any(1).all():
        raise ValueError("An annotated original instance must have a nonempty bitmap.")
    try:
        classes = torch.tensor([mapping[int(value)] for value in labels], dtype=torch.float32)
    except KeyError as error:
        raise ValueError(f"Unmapped sparse COCO category ID: {error.args[0]}") from error
    limits = torch.tensor([width, height, width, height], dtype=torch.float32)
    if (boxes < -0.0001).any() or (boxes > limits + 0.0001).any():
        raise ValueError("Original xyxy boxes lie outside the image.")
    clipped = boxes.clamp(min=0).minimum(limits)
    clipped_count = int((clipped != boxes).any(1).sum())
    boxes = clipped
    if count and ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
        raise ValueError("Original xyxy boxes must have positive width and height.")
    return boxes, classes, masks.to(torch.uint8), ids.to(torch.int64), clipped_count


def make_yolo_batch(
    images: Sequence[torch.Tensor],
    targets: Sequence[Mapping[str, Any]],
    names: Mapping[int, str] | Sequence[str],
    coco_categories: Mapping[int, Any] | Sequence[Mapping[str, Any]],
    image_size: int = 640,
    device: str | torch.device = "cpu",
    *,
    mask_downsampling: str = "nearest",
    image_resize_dtype: str = "float32",
) -> dict[str, Any]:
    """Letterbox images and retain one nearest-sampled bitmap per COCO instance.

    Image resize uses OpenCV INTER_LINEAR on RGB float32 values. The scale is
    min(S/H, S/W); dimensions use Python round, and symmetric padding uses the
    native YOLO round(half_padding +/- 0.1) rule with RGB value 114/255. Boxes use
    this same nominal scale and integer padding, then become normalized xywh.
    Mask rasterization follows MASK_SAMPLING_POLICY. Thin foreground can disappear
    through sampling; its annotation and box remain, and its ID is recorded.

    sem_masks follows pinned Ultralytics 8.4 Format(overlap_mask=False): the maximum
    class index at each sampled pixel, with zero as background. Thus class 0 has
    no semantic auxiliary foreground, although its instance target is retained.
    """
    if not images or len(images) != len(targets):
        raise ValueError("A nonempty image batch needs exactly one target per image.")
    if type(image_size) is not int or image_size < 32 or image_size % 32:
        raise ValueError("YOLO image_size must be a positive multiple of 32.")
    if mask_downsampling not in {"nearest", "linear"}:
        raise ValueError("mask_downsampling must be nearest or linear.")
    if image_resize_dtype not in {"float32", "uint8"}:
        raise ValueError("image_resize_dtype must be float32 or uint8.")
    mapping = _class_mapping(names, coco_categories)
    mask_size = image_size // MASK_RATIO
    image_tensors, class_tensors, boxes_tensors, batch_indices = [], [], [], []
    mask_tensors, semantic_tensors, annotation_ids, metadata = [], [], [], []
    for batch_index, (image, target) in enumerate(zip(images, targets, strict=True)):
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or image.shape[0] != 3
            or min(image.shape[1:]) < 1
            or image.dtype != torch.float32
            or not torch.isfinite(image).all()
            or (image < 0).any()
            or (image > 1).any()
        ):
            raise ValueError("Images must be finite RGB float32 CHW tensors in [0, 1].")
        height, width = image.shape[1:]
        boxes, classes, masks, ids, clipped_count = _target_tensors(target, height, width, mapping)
        count = len(boxes)
        scale = min(image_size / height, image_size / width)
        resized_width, resized_height = round(width * scale), round(height * scale)
        if min(resized_width, resized_height) < 1:
            raise ValueError("Letterboxing would collapse an image dimension to zero.")
        half_width = (image_size - resized_width) / 2
        half_height = (image_size - resized_height) / 2
        left, right = round(half_width - 0.1), round(half_width + 0.1)
        top, bottom = round(half_height - 0.1), round(half_height + 0.1)
        rgb = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
        if image_resize_dtype == "uint8":
            rgb = np.rint(rgb * 255).astype(np.uint8)
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        if image_resize_dtype == "uint8":
            rgb = rgb.astype(np.float32) / 255
        canvas = np.full((image_size, image_size, 3), 114 / 255, dtype=np.float32)
        canvas[top : top + resized_height, left : left + resized_width] = rgb
        image_tensors.append(torch.from_numpy(canvas).permute(2, 0, 1))
        boxes *= scale
        boxes += torch.tensor([left, top, left, top], dtype=torch.float32)
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        dimensions = boxes[:, 2:] - boxes[:, :2]
        boxes_tensors.append(torch.cat((center, dimensions), 1) / image_size)
        class_tensors.append(classes[:, None])
        batch_indices.append(torch.full((count,), batch_index, dtype=torch.int64))
        sampled_masks = np.zeros((count, mask_size, mask_size), dtype=np.uint8)
        for index, mask in enumerate(masks.numpy()):
            resized_mask = cv2.resize(
                mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
            )
            mask_canvas = np.zeros((image_size, image_size), dtype=np.uint8)
            mask_canvas[top : top + resized_height, left : left + resized_width] = resized_mask
            sampled_masks[index] = cv2.resize(
                mask_canvas,
                (mask_size, mask_size),
                interpolation=(
                    cv2.INTER_NEAREST if mask_downsampling == "nearest" else cv2.INTER_LINEAR
                ),
            )
        sampled = torch.from_numpy(sampled_masks)
        mask_tensors.append(sampled)
        if count:
            semantic = (sampled.to(torch.int64) * classes.to(torch.int64)[:, None, None]).amax(0)
        else:
            semantic = torch.zeros((mask_size, mask_size), dtype=torch.int64)
        semantic_tensors.append(semantic)
        annotation_ids.append(ids)
        vanished = ~sampled.bool().reshape(count, mask_size * mask_size).any(1)
        metadata.append(
            {
                "original_hw": [height, width],
                "resized_hw": [resized_height, resized_width],
                "scale": scale,
                "padding_ltrb": [left, top, right, bottom],
                "bbox_roundoff_clipped_count": clipped_count,
                "mask_downsampling": mask_downsampling,
                "image_resize_dtype": image_resize_dtype,
                "empty_after_mask_downsampling_annotation_ids": ids[vanished].tolist(),
            }
        )
    return {
        "img": torch.stack(image_tensors).contiguous().to(device),
        "cls": torch.cat(class_tensors).to(device),
        "bboxes": torch.cat(boxes_tensors).to(device),
        "batch_idx": torch.cat(batch_indices).to(device),
        "masks": torch.cat(mask_tensors).to(device),
        "sem_masks": torch.stack(semantic_tensors).to(device),
        "annotation_ids": torch.cat(annotation_ids).to(device),
        "resized_shape": [(image_size, image_size)] * len(images),
        "letterbox": metadata,
        "mask_sampling_policy": MASK_SAMPLING_POLICY
        if mask_downsampling == "nearest"
        else "Per-instance COCO bitmap; nearest letterbox content sampling; "
        "uint8 OpenCV INTER_LINEAR final sampling to image_size/4, as native polygon2mask. "
        "No instance removal, merging or mask repair.",
    }


def set_yolo_train_mode(network: nn.Module) -> None:
    """Enable native training outputs but freeze all BatchNorm running statistics."""
    network.train()
    for module in network.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def configure_yolo_network(yolo_wrapper: Any, cfg: Mapping[str, Any]) -> nn.Module:
    """Configure the loaded 80-class network without rebuilding any model layer.

    Import only the pinned local package after the caller installs offline guards.
    Frozen COCO checkpoint parameters are re-enabled except the first ten blocks
    and the native DFL fixed projection; all BatchNorm modules use saved running
    statistics. Loss is lazily created on the final worker device by model.loss.
    """
    from ultralytics.utils import DEFAULT_CFG_DICT

    network = yolo_wrapper.model
    if not isinstance(network, nn.Module) or not hasattr(network, "model"):
        raise ValueError("Expected an already loaded Ultralytics segmentation network.")
    _model_names(network.names)
    if getattr(network.model[-1], "nc", None) != 80:
        raise ValueError("The original 80-class segmentation head must remain unchanged.")
    settings = dict(DEFAULT_CFG_DICT)
    settings.update(
        task="segment",
        mode="train",
        epochs=3,
        overlap_mask=False,
        mask_ratio=MASK_RATIO,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    settings.update({key: value for key, value in cfg.items() if key in settings})
    if settings["overlap_mask"] is not False or settings["mask_ratio"] != MASK_RATIO:
        raise ValueError("Bitmap batches require overlap_mask=False and mask_ratio=4.")
    if type(settings["epochs"]) is not int or settings["epochs"] < 1:
        raise ValueError("Native end-to-end loss requires a positive integer epoch count.")
    if any(
        type(settings[key]) not in (int, float)
        or not math.isfinite(settings[key])
        or settings[key] < 0
        for key in ("box", "cls", "dfl")
    ):
        raise ValueError("Native loss gains must be finite and nonnegative.")
    freeze_blocks = cfg.get("freeze_backbone_blocks", cfg.get("freeze_blocks", 10))
    if type(freeze_blocks) is not int or not 0 <= freeze_blocks < len(network.model):
        raise ValueError("freeze_backbone_blocks must leave at least the native head trainable.")
    network.args = SimpleNamespace(**settings)
    network.criterion = None
    for parameter in network.parameters():
        parameter.requires_grad_(True)
    for block in list(network.model)[:freeze_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad_(False)
    for name, parameter in network.named_parameters():
        if ".dfl." in name:
            parameter.requires_grad_(False)
    set_yolo_train_mode(network)
    LOGGER.info("Configured unchanged COCO80 YOLO network; frozen blocks=%d", freeze_blocks)
    return network
