"""Bounded, offline supervised fine-tuning of the four existing COCO checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.verification import sha256_file

LOGGER = logging.getLogger(__name__)


def read_settings(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open("rb") as handle:
        cfg = tomllib.load(handle)
    cfg["model_config"] = (path.parent / cfg["model_config"]).resolve()
    cfg["output"] = (path.parent / cfg["output"]).resolve()
    with cfg["model_config"].open("rb") as handle:
        models = tomllib.load(handle)["models"]
    for model in models:
        model["checkpoint"] = (cfg["model_config"].parent / model["checkpoint"]).resolve()
    cfg["models"] = models
    cfg["source"] = path
    for field in ("epochs", "batch_size", "image_size", "threads"):
        if type(cfg[field]) is not int or cfg[field] < 1:
            raise Phase1Error(f"Invalid training setting: {field}")
    return cfg


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def deny_network() -> None:
    """Make accidental downloads or service calls fail closed in model workers."""
    import socket

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network is disabled for this local training experiment.")

    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden


def freeze_batch_norm(network: Any) -> None:
    import torch

    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_digests(network: Any) -> dict[str, str]:
    import hashlib

    return {
        name: hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, parameter in network.named_parameters()
    }


def evaluate(
    network: Any,
    yolo: Any,
    model_cfg: dict[str, Any],
    subset: Path,
    output: Path,
    cfg: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from pycocotools.coco import COCO

    from animal_eye_metrology.coco import ImageRecord
    from animal_eye_metrology.phase2_pilot import clip_box_roundoff
    from animal_eye_metrology.segmentation import evaluate_predictions, normalize_predictions
    from animal_eye_metrology.training_data import FrozenCocoDataset

    output.mkdir(exist_ok=False)
    device = cfg.get("evaluation_device", "cpu")
    frozen = FrozenCocoDataset(subset)
    content = json.loads(subset.read_text())
    records = {row["id"]: row for row in content["images"]}
    categories = {row["id"]: row for row in content["categories"]}
    animals = {key for key, row in categories.items() if row["supercategory"] == "animal"}
    predictor = None
    if yolo is not None:
        from ultralytics import YOLO

        predictor = YOLO(str(model_cfg["checkpoint"]), task="segment")
        # Native prediction may fuse Conv/BN. It must never mutate the training network.
        predictor.model = copy.deepcopy(network).to(device).eval()
        predictor.model.args = dict(vars(network.args))
        predictor.predictor = None
        names = predictor.names
    else:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

        names = dict(enumerate(MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"]))
        evaluation_network = copy.deepcopy(network).to(device).eval()
    mapping = {
        index: category
        for index, name in names.items()
        for category in animals
        if categories[category]["name"] == name
    }
    if len(mapping) != 10:
        raise Phase1Error("Evaluation requires all ten original COCO animal class mappings.")
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(frozen)):
            image, target = frozen[index]
            key = int(target["image_id"].reshape(-1)[0])
            row = records[key]
            record = ImageRecord(key, row["file_name"], row["width"], row["height"])
            if predictor is not None:
                pixels = image.mul(255).round().byte().permute(1, 2, 0).numpy()
                prediction = predictor.predict(
                    Image.fromarray(pixels),
                    imgsz=cfg["image_size"],
                    device=device,
                    conf=cfg["confidence_floor"],
                    iou=0.7,
                    retina_masks=True,
                    rect=False,
                    max_det=cfg["max_detections"],
                    half=False,
                    augment=False,
                    save=False,
                    verbose=False,
                )[0]
                boxes = prediction.boxes.data.cpu().numpy()
                masks = (
                    prediction.masks.data.cpu().numpy() if prediction.masks is not None else None
                )
                torch.set_num_threads(cfg["threads"])
            else:
                prediction = evaluation_network([image.to(device)])[0]
                boxes = (
                    torch.cat(
                        [
                            prediction["boxes"],
                            prediction["scores"][:, None],
                            prediction["labels"][:, None],
                        ],
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
                masks = prediction["masks"][:, 0].cpu().numpy()
            boxes, _ = clip_box_roundoff(np.asarray(boxes), record.width, record.height)
            predictions.extend(normalize_predictions(record, boxes, masks, mapping))
    inference_seconds = time.perf_counter() - started
    truth = COCO(str(subset))
    metrics, matches = evaluate_predictions(truth, predictions, sorted(records), sorted(animals))
    metrics["scope"] = "COCO_train2017_internal_split_not_unseen_pretraining_test"
    metrics["inference_seconds"] = inference_seconds
    metrics["evaluation_device"] = device
    write_json(output / "metrics.json", metrics)
    write_json(output / "predictions.json", predictions)
    with (output / "matches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matches[0]) if matches else ["image_id"])
        writer.writeheader()
        writer.writerows(matches)
    del predictor, frozen
    if yolo is None:
        del evaluation_network
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return metrics


def run_model(cfg: dict[str, Any], model_name: str, device: str) -> None:
    cache = cfg["output"] / "cache" / model_name
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        YOLO_CONFIG_DIR=str(cache),
        YOLO_OFFLINE="true",
        YOLO_AUTOINSTALL="false",
        MPLCONFIGDIR=str(cache / "matplotlib"),
        WANDB_DISABLED="true",
        PYTORCH_ENABLE_MPS_FALLBACK="1",
    )
    deny_network()
    import numpy as np
    import torch

    from animal_eye_metrology.training_data import FrozenCocoDataset

    model_cfg = next(row for row in cfg["models"] if row["name"] == model_name)
    if sha256_file(model_cfg["checkpoint"]) != model_cfg["sha256"]:
        raise Phase1Error("Original checkpoint differs from its pinned digest.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise Phase1Error("MPS is unavailable in this process.")
    output = cfg["output"] / model_name
    output.mkdir(exist_ok=False)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yolo = None
    if model_cfg["backend"] == "ultralytics":
        import ultralytics
        from ultralytics import YOLO

        from animal_eye_metrology.training_yolo import (
            configure_yolo_network,
            make_yolo_batch,
            set_yolo_train_mode,
        )

        if ultralytics.__version__ != model_cfg["package_version"]:
            raise Phase1Error("YOLO package differs from the pinned model version.")
        yolo = YOLO(str(model_cfg["checkpoint"]), task="segment")
        network = configure_yolo_network(
            yolo, cfg | {"freeze_backbone_blocks": cfg["freeze_yolo_blocks"]}
        ).to(device)
    else:
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

        network = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=91,
            min_size=cfg["image_size"],
            max_size=cfg["maskrcnn_max_size"],
            box_score_thresh=cfg["confidence_floor"],
            box_detections_per_img=cfg["max_detections"],
        )
        network.load_state_dict(
            torch.load(model_cfg["checkpoint"], map_location="cpu", weights_only=True), strict=True
        )
        for parameter in network.backbone.parameters():
            parameter.requires_grad_(False)
        network.to(device)
    before = parameter_digests(network)
    trainable = {name: p.numel() for name, p in network.named_parameters() if p.requires_grad}
    optimizer = torch.optim.SGD(
        [p for p in network.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    data = cfg["output"] / "data"
    fitting = FrozenCocoDataset(data / "fitting.json")
    source = json.loads((data / "fitting.json").read_text())
    write_json(
        output / "run_manifest.json",
        {
            "status": "running",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model": model_name,
            "device": device,
            "evaluation_device": cfg.get("evaluation_device", "cpu"),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "source_checkpoint_sha256": model_cfg["sha256"],
            "config_sha256": sha256_file(cfg["source"]),
            "trainable_parameters": trainable,
            "training_policy": "native supervised losses; frozen backbone and all BN running stats",
            "bitmap_policy": "COCO annToMask; all classes and original instance identities",
            "execution_argv": [sys.executable, *sys.argv],
            "seed": cfg["seed"],
            "epochs": cfg["epochs"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "no_augmentation": True,
            "official_val_used": False,
        },
    )
    baseline = evaluate(
        network, yolo, model_cfg, data / "holdout.json", output / "baseline_holdout", cfg, device
    )
    steps = 0
    best_ap = -1.0
    epochs = []
    training_started = time.perf_counter()
    with (output / "steps.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(cfg["epochs"]):
            order = list(range(len(fitting)))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            losses = []
            for offset in range(0, len(order), cfg["batch_size"]):
                examples = [fitting[index] for index in order[offset : offset + cfg["batch_size"]]]
                images = [example[0] for example in examples]
                targets = [example[1] for example in examples]
                optimizer.zero_grad(set_to_none=True)
                if yolo is not None:
                    set_yolo_train_mode(network)
                    batch = make_yolo_batch(
                        images,
                        targets,
                        network.names,
                        source["categories"],
                        cfg["image_size"],
                        device,
                    )
                    loss_vector, components = network(batch)
                    loss = loss_vector.sum() / len(images)
                    component_record = components.detach().cpu().tolist()
                else:
                    network.train()
                    freeze_batch_norm(network)
                    moved = [
                        {key: value.to(device) for key, value in row.items()} for row in targets
                    ]
                    native = network([image.to(device) for image in images], moved)
                    loss = sum(native.values())
                    component_record = {
                        key: float(value.detach().cpu()) for key, value in native.items()
                    }
                if not bool(torch.isfinite(loss)):
                    raise Phase1Error("Nonfinite training loss; refusing an optimizer update.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in network.parameters() if p.requires_grad],
                    cfg["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                steps += 1
                scalar = float(loss.detach().cpu())
                losses.append(scalar)
                record = {
                    "epoch": epoch + 1,
                    "optimizer_step": steps,
                    "image_ids": [int(row["image_id"].reshape(-1)[0]) for row in targets],
                    "loss": scalar,
                    "components": component_record,
                    "gradient_norm_before_clip": float(norm.detach().cpu()),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if steps % 16 == 0:
                    LOGGER.info(
                        json.dumps({"event": "training_progress", "model": model_name, **record})
                    )
            development = evaluate(
                network,
                yolo,
                model_cfg,
                data / "development.json",
                output / f"development_epoch_{epoch + 1}",
                cfg,
                device,
            )
            checkpoint = {
                "model_state": {
                    key: value.detach().cpu() for key, value in network.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "optimizer_steps": steps,
                "source_checkpoint_sha256": model_cfg["sha256"],
            }
            torch.save(checkpoint, output / "last.pt")
            ap = development["coco_segm"]["mask_ap"]
            if ap > best_ap:
                best_ap = ap
                torch.save(checkpoint, output / "best.pt")
            epochs.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": sum(losses) / len(losses),
                    "development_mask_ap": ap,
                }
            )
            write_json(output / "epochs.json", epochs)
            LOGGER.info(json.dumps({"event": "epoch_complete", "model": model_name, **epochs[-1]}))
            criterion = getattr(network, "criterion", None)
            if criterion is not None and hasattr(criterion, "update"):
                criterion.update()
    training_wall_seconds = time.perf_counter() - training_started
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    network.load_state_dict(best["model_state"], strict=True)
    after = parameter_digests(network)
    changed = [name for name, digest in before.items() if after[name] != digest]
    if not changed or not steps:
        raise Phase1Error("Training produced no verified optimizer steps or parameter changes.")
    final = evaluate(
        network, yolo, model_cfg, data / "holdout.json", output / "finetuned_holdout", cfg, device
    )
    result = {
        "status": "complete",
        "model": model_name,
        "device": device,
        "epochs": epochs,
        "optimizer_steps": steps,
        "best_epoch": best["epoch"],
        "changed_parameter_count": len(changed),
        "changed_parameter_names": changed,
        "baseline_holdout_mask_ap": baseline["coco_segm"]["mask_ap"],
        "finetuned_holdout_mask_ap": final["coco_segm"]["mask_ap"],
        "training_wall_seconds_including_development": training_wall_seconds,
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "source_checkpoint_unchanged": sha256_file(model_cfg["checkpoint"]) == model_cfg["sha256"],
        "scope": "bounded COCO_train2017 pilot; not full-data training "
        "or unseen pretraining evaluation",
    }
    write_json(output / "result.json", result)
    LOGGER.info(json.dumps({"event": "model_complete", **result}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "all", "summarize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("cpu", "mps"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = read_settings(args.config)
    root = Path.cwd().resolve()
    if args.action == "prepare":
        from animal_eye_metrology.training_data import prepare_training_data

        cfg["output"].mkdir(parents=True, exist_ok=True)
        path = prepare_training_data(
            repository_root=root,
            output_directory=cfg["output"] / "data",
            fitting_size=cfg["fitting_size"],
            development_size=cfg["development_size"],
            holdout_size=cfg["holdout_size"],
            seed=cfg["seed"],
            split_seed=cfg["split_seed"],
            candidate_limit=cfg["candidate_limit"],
        )
        LOGGER.info(json.dumps({"event": "training_data_prepared", "manifest": str(path)}))
    elif args.action == "run":
        run_model(cfg, args.model, args.device or cfg["device"])
    elif args.action == "all":
        for model in cfg["models"]:
            env = os.environ.copy()
            if model["package_version"] == "8.4.0":
                env["PYTHONPATH"] = str(root / "outputs/phase2/pilot100/ultralytics_8_4_0")
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            command = [
                sys.executable,
                "-m",
                "animal_eye_metrology.training",
                "run",
                "--config",
                str(cfg["source"]),
                "--model",
                model["name"],
                "--device",
                args.device or cfg["device"],
            ]
            LOGGER.info(json.dumps({"event": "starting_model", "model": model["name"]}))
            with (cfg["output"] / f"{model['name']}.log").open("x") as handle:
                completed = subprocess.run(
                    command, env=env, stdout=handle, stderr=subprocess.STDOUT
                )
            LOGGER.info(
                json.dumps(
                    {
                        "event": "model_process_exited",
                        "model": model["name"],
                        "returncode": completed.returncode,
                    }
                )
            )
            if completed.returncode:
                raise Phase1Error(f"Model worker failed; inspect {model['name']}.log.")
    else:
        results = [
            json.loads((cfg["output"] / model["name"] / "result.json").read_text())
            for model in cfg["models"]
        ]
        write_json(cfg["output"] / "comparison.json", results)
        fields = (
            "model",
            "device",
            "optimizer_steps",
            "best_epoch",
            "baseline_holdout_mask_ap",
            "finetuned_holdout_mask_ap",
            "changed_parameter_count",
        )
        with (cfg["output"] / "comparison.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        LOGGER.info(json.dumps({"event": "training_comparison_saved", "models": len(results)}))


if __name__ == "__main__":
    main()
