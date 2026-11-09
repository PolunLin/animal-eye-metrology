"""Development-only YOLO retraining and a separately frozen fresh evaluation stage."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.research.training import (
    deny_network,
    evaluate,
    parameter_digests,
    read_settings,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def settings(path: Path) -> dict[str, Any]:
    cfg = read_settings(path)
    for field in ("data_directory", "previous_output"):
        cfg[field] = (cfg["source"].parent / cfg[field]).resolve()
    cfg["models"] = [row for row in cfg["models"] if row["name"] in cfg["model_names"]]
    if len(cfg["models"]) != 3 or any(row["backend"] != "ultralytics" for row in cfg["models"]):
        raise Phase1Error("Recovery is restricted to the three approved original YOLO models.")
    return cfg


def initialize(cfg: dict[str, Any], name: str) -> tuple[Any, dict[str, Any]]:
    cache = cfg["output"] / "cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        YOLO_CONFIG_DIR=str(cache),
        YOLO_OFFLINE="true",
        YOLO_AUTOINSTALL="false",
        MPLCONFIGDIR=str(cache / "matplotlib"),
        WANDB_DISABLED="true",
    )
    deny_network()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    from animal_eye_metrology.models.training_yolo import configure_yolo_network

    model_cfg = next(row for row in cfg["models"] if row["name"] == name)
    if sha256_file(model_cfg["checkpoint"]) != model_cfg["sha256"]:
        raise Phase1Error("Original checkpoint changed.")
    if ultralytics.__version__ != model_cfg["package_version"]:
        raise Phase1Error("Use the exact model-specific isolated Ultralytics package.")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wrapper = YOLO(str(model_cfg["checkpoint"]), task="segment")
    configure_yolo_network(wrapper, cfg | {"freeze_backbone_blocks": cfg["freeze_yolo_blocks"]})
    wrapper.model.cpu()
    return wrapper, model_cfg


def train(cfg: dict[str, Any], name: str) -> None:
    wrapper, model_cfg = initialize(cfg, name)
    import torch

    from animal_eye_metrology.data.training_data import FrozenCocoDataset
    from animal_eye_metrology.models.training_yolo import make_yolo_batch, set_yolo_train_mode

    network = wrapper.model
    output = cfg["output"] / name
    output.mkdir(parents=True, exist_ok=False)
    data = cfg["data_directory"]
    fitting = FrozenCocoDataset(data / "fitting.json")
    categories = fitting.coco.dataset["categories"]
    original_parameters = parameter_digests(network)
    optimizer = torch.optim.SGD(
        [parameter for parameter in network.parameters() if parameter.requires_grad],
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    baseline = evaluate(
        network,
        wrapper,
        model_cfg,
        data / "development.json",
        output / "baseline_development",
        cfg,
        "cpu",
    )
    baseline_ap = baseline["coco_segm"]["mask_ap"]
    sources = {
        cfg["source"],
        cfg["model_config"],
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(make_yolo_batch)),
        data / "manifest.json",
        data / "fitting.json",
        data / "development.json",
    }
    source_digests = {str(path): sha256_file(path) for path in sorted(sources)}
    write_json(
        output / "run_manifest.json",
        {
            "status": "training",
            "model": name,
            "source_sha256": source_digests,
            "source_checkpoint_sha256": model_cfg["sha256"],
            "seed": cfg["seed"],
            "device": "cpu",
            "epochs": cfg["epochs"],
            "learning_rate": cfg["learning_rate"],
            "mask_downsampling": cfg["mask_downsampling"],
            "image_resize_dtype": cfg["image_resize_dtype"],
            "freeze_policy": "prefix blocks 0..9 plus fixed DFL; all BN running statistics frozen",
            "baseline_development_mask_ap": baseline_ap,
            "fresh_holdout_accessed_during_training": False,
        },
    )
    steps, best_ap, best_epoch = 0, -1.0, 0
    epochs = []
    started = time.perf_counter()
    with (output / "steps.jsonl").open("x") as log:
        for epoch in range(cfg["epochs"]):
            order = list(range(len(fitting)))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            losses, clipped = [], 0
            for offset in range(0, len(order), cfg["batch_size"]):
                examples = [fitting[index] for index in order[offset : offset + cfg["batch_size"]]]
                images, targets = [row[0] for row in examples], [row[1] for row in examples]
                set_yolo_train_mode(network)
                optimizer.zero_grad(set_to_none=True)
                batch = make_yolo_batch(
                    images,
                    targets,
                    network.names,
                    categories,
                    cfg["image_size"],
                    "cpu",
                    mask_downsampling=cfg["mask_downsampling"],
                    image_resize_dtype=cfg["image_resize_dtype"],
                )
                vector, components = network(batch)
                loss = vector.sum() / len(images)
                if not bool(torch.isfinite(loss)):
                    raise Phase1Error("Nonfinite loss; no update performed.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in network.parameters() if parameter.requires_grad],
                    cfg["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                steps += 1
                clipped += int(float(norm) > cfg["gradient_clip_norm"])
                losses.append(float(loss.detach()))
                row = {
                    "epoch": epoch + 1,
                    "optimizer_step": steps,
                    "image_ids": [int(target["image_id"]) for target in targets],
                    "loss": losses[-1],
                    "gradient_norm_before_clip": float(norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "components": components.detach().tolist(),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                if steps % 32 == 0:
                    LOGGER.info(json.dumps({"event": "retraining_progress", "model": name, **row}))
            development = evaluate(
                network,
                wrapper,
                model_cfg,
                data / "development.json",
                output / f"development_epoch_{epoch + 1}",
                cfg,
                "cpu",
            )
            ap = development["coco_segm"]["mask_ap"]
            state = {
                "model_state": {
                    key: value.detach().cpu().clone() for key, value in network.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "optimizer_steps": steps,
                "source_checkpoint_sha256": model_cfg["sha256"],
            }
            torch.save(state, output / "last.pt")
            if ap > best_ap:
                best_ap, best_epoch = ap, epoch + 1
                torch.save(state, output / "best.pt")
            epochs.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": sum(losses) / len(losses),
                    "development_mask_ap": ap,
                    "clipped_batch_fraction": clipped / len(losses),
                }
            )
            write_json(output / "epochs.json", epochs)
            LOGGER.info(
                json.dumps({"event": "retraining_epoch_complete", "model": name, **epochs[-1]})
            )
            criterion = getattr(network, "criterion", None)
            if criterion is not None and hasattr(criterion, "update"):
                criterion.update()
    if any(sha256_file(Path(path)) != digest for path, digest in source_digests.items()):
        raise Phase1Error("A training source changed during the experiment.")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    network.load_state_dict(best["model_state"], strict=True)
    after = parameter_digests(network)
    changed = [key for key, digest in original_parameters.items() if after[key] != digest]
    write_json(
        output / "result.json",
        {
            "status": "trained_pending_fresh_evaluation",
            "model": name,
            "device": "cpu",
            "optimizer_steps": steps,
            "best_epoch": best_epoch,
            "epochs": epochs,
            "baseline_development_mask_ap": baseline_ap,
            "best_development_mask_ap": best_ap,
            "beats_pretrained_on_development": best_ap > baseline_ap,
            "recommended_candidate_from_development": (
                "finetuned" if best_ap > baseline_ap else "pretrained"
            ),
            "best_pt_policy": (
                "Best trained epoch; pretrained baseline remains separately eligible."
            ),
            "changed_parameter_count": len(changed),
            "best_checkpoint_sha256": sha256_file(output / "best.pt"),
            "source_checkpoint_unchanged": sha256_file(model_cfg["checkpoint"])
            == model_cfg["sha256"],
            "training_wall_seconds_including_development": time.perf_counter() - started,
            "fresh_holdout_used_for_parameter_selection": False,
        },
    )


def _verified_training_sources(cfg: dict[str, Any]) -> dict[str, str]:
    """Bind the first assessment to the inputs recorded by every training run.

    Code snapshots remain historical provenance: only configuration and data
    identity must match here, so maintenance does not invalidate prior training.
    """
    data = cfg["data_directory"]
    paths = (
        cfg["source"],
        cfg["model_config"],
        data / "manifest.json",
        data / "fitting.json",
        data / "development.json",
    )
    try:
        sources = {str(path): sha256_file(path) for path in paths}
        for model in cfg["models"]:
            path = cfg["output"] / model["name"] / "run_manifest.json"
            manifest = json.loads(path.read_text())
            recorded = manifest["source_sha256"]
            if (
                manifest["model"] != model["name"]
                or manifest["source_checkpoint_sha256"] != model["sha256"]
                or not isinstance(recorded, dict)
                or any(recorded.get(str(source)) != sources[str(source)] for source in paths)
            ):
                raise Phase1Error(
                    f"Training configuration or data differs from the recorded run: {model['name']}"
                )
            sources[str(path)] = sha256_file(path)
        dataset = json.loads((data / "manifest.json").read_text())
        if dataset["status"] != "frozen":
            raise Phase1Error("Fresh evaluation requires the recorded frozen dataset.")
        for split in ("fitting", "development", "holdout"):
            path = data / f"{split}.json"
            digest = sha256_file(path)
            if dataset["artifact_sha256"].get(path.name) != digest:
                raise Phase1Error(f"Frozen evaluation dataset changed: {path.name}")
            sources[str(path)] = digest
    except Phase1Error:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise Phase1Error("Missing or invalid training provenance for fresh evaluation.") from error
    return sources


def _verify_frozen_sources(selection: dict[str, Any]) -> None:
    """Verify new source-bound freezes while preserving legacy freeze compatibility."""
    if "source_sha256" not in selection:
        return
    try:
        sources = selection["source_sha256"]
        if not isinstance(sources, dict) or not sources:
            raise ValueError("A source-bound freeze requires recorded input digests.")
        for path, digest in sources.items():
            if sha256_file(Path(path)) != digest:
                raise Phase1Error(f"Frozen evaluation source changed: {path}")
    except Phase1Error:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise Phase1Error("Missing or invalid frozen evaluation provenance.") from error


def finalize(cfg: dict[str, Any], name: str) -> None:
    # Freeze every model selection before the first new assessment prediction.
    freeze = cfg["output"] / "selection_frozen.json"
    if not freeze.exists():
        selections = {}
        for row in cfg["models"]:
            result = json.loads((cfg["output"] / row["name"] / "result.json").read_text())
            if result["status"] != "trained_pending_fresh_evaluation":
                raise Phase1Error(
                    "All three development-only runs must finish before fresh evaluation."
                )
            selections[row["name"]] = {
                key: result[key]
                for key in (
                    "best_epoch",
                    "best_checkpoint_sha256",
                    "best_development_mask_ap",
                    "recommended_candidate_from_development",
                )
            }
        sources = _verified_training_sources(cfg)
        with freeze.open("x") as handle:
            json.dump(
                {
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "config_sha256": sha256_file(cfg["source"]),
                    "source_sha256": sources,
                    "selections": selections,
                },
                handle,
                indent=2,
            )
    selection = json.loads(freeze.read_text())
    if selection["config_sha256"] != sha256_file(cfg["source"]):
        raise Phase1Error("Settings changed after the development-only selection freeze.")
    _verify_frozen_sources(selection)
    wrapper, model_cfg = initialize(cfg, name)
    import torch

    output, data = cfg["output"] / name, cfg["data_directory"]
    result = json.loads((output / "result.json").read_text())
    if sha256_file(output / "best.pt") != selection["selections"][name]["best_checkpoint_sha256"]:
        raise Phase1Error("Selected checkpoint changed after freezing.")
    metrics = {}
    for label, checkpoint in (
        ("pretrained", None),
        ("previous_v1", cfg["previous_output"] / name / "best.pt"),
        ("retrained_v2", output / "best.pt"),
    ):
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            wrapper.model.load_state_dict(state["model_state"], strict=True)
        metrics[label] = evaluate(
            wrapper.model,
            wrapper,
            model_cfg,
            data / "holdout.json",
            output / f"fresh_holdout_{label}",
            cfg,
            "cpu",
        )["coco_segm"]["mask_ap"]
    result.update(
        status="complete",
        fresh_holdout_mask_ap=metrics,
        baseline_holdout_mask_ap=metrics["pretrained"],
        finetuned_holdout_mask_ap=metrics["retrained_v2"],
        scope="fresh project assessment from unused COCO train groups; "
        "upstream overlap remains possible",
    )
    write_json(output / "result.json", result)
    LOGGER.info(json.dumps({"event": "fresh_evaluation_complete", **result}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = settings(args.config)
    (train if args.action == "train" else finalize)(cfg, args.model)


if __name__ == "__main__":
    main()
