"""Synthetic bitmap/coordinate checks, never a model-accuracy experiment."""

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from animal_eye_metrology.models.training_yolo import (  # noqa: E402
    configure_yolo_network,
    make_yolo_batch,
    set_yolo_train_mode,
)


def test_legacy_adapter_import_shares_the_canonical_module():
    assert importlib.import_module("animal_eye_metrology.training_yolo") is importlib.import_module(
        "animal_eye_metrology.models.training_yolo"
    )


def _names():
    names = {index: f"class_{index}" for index in range(80)}
    names[0], names[16], names[21] = "person", "dog", "bear"
    return names


def _target(boxes, masks, labels=None, ids=None):
    count = len(boxes)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
        "labels": torch.tensor(labels or [18] * count, dtype=torch.int64),
        "masks": masks,
        "annotation_id": torch.tensor(ids or list(range(count)), dtype=torch.int64),
    }


def test_letterbox_preserves_rgb_box_alignment_and_original_inputs():
    image = torch.zeros((3, 32, 64), dtype=torch.float32)
    image[0] = 1  # RGB red must remain the first channel.
    masks = torch.zeros((1, 32, 64), dtype=torch.uint8)
    masks[0, 8:24, 16:48] = 1
    target = _target([[16, 8, 48, 24]], masks, ids=[71])
    before = target["boxes"].clone()
    batch = make_yolo_batch([image], [target], _names(), {18: {"name": "dog"}}, image_size=64)
    assert batch["img"].shape == (1, 3, 64, 64)
    assert batch["img"][0, :, 20, 10].tolist() == [1, 0, 0]
    assert batch["img"][0, :, 0, 0].tolist() == pytest.approx([114 / 255] * 3)
    assert batch["bboxes"][0].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.25])
    assert batch["cls"].tolist() == [[16]]
    assert batch["batch_idx"].tolist() == [0]
    assert batch["annotation_ids"].tolist() == [71]
    expected = torch.zeros((16, 16), dtype=torch.uint8)
    expected[6:10, 4:12] = 1
    assert torch.equal(batch["masks"][0], expected)
    assert batch["letterbox"][0]["padding_ltrb"] == [0, 16, 0, 16]
    assert batch["letterbox"][0]["empty_after_mask_downsampling_annotation_ids"] == []
    assert torch.equal(target["boxes"], before)
    assert torch.equal(target["masks"], masks)


def test_disconnected_and_overlapping_instances_keep_identity_and_sparse_mapping():
    masks = torch.zeros((3, 64, 64), dtype=torch.uint8)
    masks[0, 4:12, 4:12] = 1
    masks[0, 40:48, 40:48] = 1  # Two components remain one annotation.
    masks[1, 4:12, 4:12] = 1  # A second instance may occupy the same pixels.
    masks[2, 16:24, 16:24] = 1
    target = _target(
        [[4, 4, 48, 48], [4, 4, 12, 12], [16, 16, 24, 24]],
        masks,
        labels=[18, 23, 1],
        ids=[101, 102, 103],
    )
    categories = [
        {"id": 1, "name": "person"},
        {"id": 18, "name": "dog"},
        {"id": 23, "name": "bear"},
    ]
    batch = make_yolo_batch([torch.zeros(3, 64, 64)], [target], _names(), categories, 64)
    assert batch["masks"].shape == (3, 16, 16)
    assert batch["masks"][0].sum() == 8
    assert batch["masks"][1].sum() == 4
    assert batch["masks"][0, 1, 1] == batch["masks"][1, 1, 1] == 1
    assert batch["cls"].flatten().tolist() == [16, 21, 0]
    assert batch["annotation_ids"].tolist() == [101, 102, 103]
    assert batch["sem_masks"][0, 1, 1] == 21
    assert batch["sem_masks"][0, 10, 10] == 16
    assert batch["sem_masks"][0, 4, 4] == 0  # Native class-zero semantic convention.
    assert batch["masks"][2, 4, 4] == 1  # Person instance remains intact.


def test_background_images_and_sampled_away_masks_do_not_change_instance_denominator():
    masks = torch.zeros((1, 64, 64), dtype=torch.uint8)
    masks[0, 1, 1] = 1  # Nearest sample sites are 0,4,8,..., so this pixel vanishes.
    targets = [
        _target([], torch.zeros((0, 32, 64), dtype=torch.uint8)),
        _target([[1, 1, 2, 2]], masks, ids=[93]),
    ]
    batch = make_yolo_batch(
        [torch.zeros(3, 32, 64), torch.zeros(3, 64, 64)],
        targets,
        _names(),
        {18: SimpleNamespace(name="dog")},
        64,
    )
    assert batch["masks"].shape == (1, 16, 16)
    assert batch["masks"].sum() == 0
    assert batch["batch_idx"].tolist() == [1]
    assert batch["letterbox"][1]["empty_after_mask_downsampling_annotation_ids"] == [93]
    assert batch["annotation_ids"].tolist() == [93]
    empty = make_yolo_batch(
        [torch.zeros(3, 32, 64)], [targets[0]], _names(), {18: {"name": "dog"}}, 64
    )
    assert empty["bboxes"].shape == (0, 4)
    assert empty["cls"].shape == (0, 1)
    assert empty["masks"].shape == (0, 16, 16)
    assert empty["sem_masks"].shape == (1, 16, 16)


def test_invalid_targets_and_changed_head_are_rejected():
    image = torch.zeros(3, 32, 32)
    masks = torch.ones((1, 32, 32), dtype=torch.uint8)
    target = _target([[0, 0, 32, 32]], masks)
    args = ([image], [target], _names(), {18: {"name": "dog"}})
    with pytest.raises(ValueError, match="80 unique"):
        make_yolo_batch([image], [target], {0: "dog"}, {18: {"name": "dog"}})
    with pytest.raises(ValueError, match="multiple of 32"):
        make_yolo_batch(*args, image_size=65)
    with pytest.raises(ValueError, match="outside"):
        make_yolo_batch([image], [target | {"boxes": torch.tensor([[0, 0, 33, 32]])}], *args[2:])
    with pytest.raises(ValueError, match="binary"):
        make_yolo_batch([image], [target | {"masks": masks * 255}], *args[2:])
    with pytest.raises(ValueError, match="Unmapped"):
        make_yolo_batch([image], [target | {"labels": torch.tensor([999])}], *args[2:])
    with pytest.raises(ValueError, match="align"):
        make_yolo_batch([image], [target | {"masks": masks[:, :-1]}], *args[2:])


def test_configuration_keeps_architecture_weights_and_freezes_backbone_and_bn(monkeypatch):
    # Stub only the defaults import: no Ultralytics import, checkpoint, or network call.
    import sys

    monkeypatch.setitem(
        sys.modules,
        "ultralytics.utils",
        SimpleNamespace(DEFAULT_CFG_DICT={"epochs": 100, "box": 1, "cls": 1, "dfl": 1}),
    )
    network = torch.nn.Module()
    blocks = [torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3))]
    blocks += [torch.nn.Conv2d(3, 3, 1) for _ in range(10)]
    head = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 80, 1))
    head.nc = 80
    network.model = torch.nn.Sequential(*blocks, head)
    network.names = _names()
    network.criterion = object()
    original_head = network.model[-1]
    saved = {name: value.clone() for name, value in network.state_dict().items()}
    for parameter in network.parameters():
        parameter.requires_grad_(False)  # Stripped pretrained checkpoints freeze parameters.
    assert configure_yolo_network(SimpleNamespace(model=network), {"epochs": 3}) is network
    assert network.model[-1] is original_head
    assert network.args.epochs == 3
    assert network.args.overlap_mask is False
    assert network.args.mask_ratio == 4
    assert network.args.box == 7.5
    assert network.args.cls == 0.5
    assert network.args.dfl == 1.5
    assert network.criterion is None
    assert not any(
        p.requires_grad for block in list(network.model)[:10] for p in block.parameters()
    )
    assert all(p.requires_grad for p in network.model[-1].parameters())
    assert all(torch.equal(value, saved[name]) for name, value in network.state_dict().items())
    assert not any(
        module.training for module in network.modules() if isinstance(module, torch.nn.BatchNorm2d)
    )
    network.train()
    set_yolo_train_mode(network)
    assert network.training
    assert not network.model[-1][0].training
    with pytest.raises(ValueError, match="overlap_mask"):
        configure_yolo_network(SimpleNamespace(model=network), {"overlap_mask": True})
