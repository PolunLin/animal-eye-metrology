"""Controlled CPU learning-rate diagnosis with the frozen first-pilot bitmap recipe."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.research import training

LR_CANDIDATES = {"legacy_1e3": 0.001, "lr_1e4": 0.0001, "lr_1e5": 0.00001}


def guard_lr_only(candidates: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject an alleged LR-only comparison if any other setting differs."""
    if len(candidates) < 2:
        raise ValueError("An LR-only comparison requires at least two candidates.")
    common = None
    for candidate in candidates.values():
        rate = candidate.get("learning_rate")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Every learning rate must be positive and finite.")
        other = {key: value for key, value in candidate.items() if key != "learning_rate"}
        if common is None:
            common = other
        elif other != common:
            raise ValueError("LR-only candidates differ in a setting other than learning_rate.")


def relative_parameter_drift(
    initial: Mapping[str, Any], current: Mapping[str, Any], groups: Mapping[str, list[str]]
) -> dict[str, dict[str, Any]]:
    """Compute pooled L2 drift in float64; a zero reference norm gives no ratio."""
    import torch

    if initial.keys() != current.keys():
        raise ValueError("Drift states need identical parameter keys.")
    result = {}
    for group, names in groups.items():
        if len(set(names)) != len(names) or set(names) - initial.keys():
            raise ValueError("Parameter groups must contain unique known names.")
        delta_squared = reference_squared = 0.0
        for name in names:
            left, right = initial[name].detach().cpu(), current[name].detach().cpu()
            if left.shape != right.shape or not torch.isfinite(right).all():
                raise ValueError("Parameter drift requires aligned finite tensors.")
            left, right = left.double(), right.double()
            delta_squared += float((right - left).square().sum())
            reference_squared += float(left.square().sum())
        absolute, reference = math.sqrt(delta_squared), math.sqrt(reference_squared)
        result[group] = {
            "parameter_tensors": len(names),
            "absolute_l2": absolute,
            "reference_l2": reference,
            "relative_l2": absolute / reference if reference else None,
        }
    return result


def guard_data_paths(fitting: Path, development: Path) -> None:
    """The diagnosis has no API for reading or evaluating a holdout subset."""
    if (
        fitting.name != "fitting.json"
        or development.name != "development.json"
        or fitting.parent.resolve() != development.parent.resolve()
    ):
        raise ValueError("The probe accepts only the paired fitting/development subsets.")


def _json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _runtime(cfg: Mapping[str, Any], output: Path) -> Any:
    os.environ.update(
        YOLO_OFFLINE="true",
        YOLO_AUTOINSTALL="false",
        WANDB_DISABLED="true",
        YOLO_CONFIG_DIR=str(output / "cache"),
        MPLCONFIGDIR=str(output / "cache/matplotlib"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    training.deny_network()
    import torch
    import ultralytics
    from ultralytics.utils import torch_utils

    if ultralytics.__version__ != "8.3.230" or cfg["device"] != "cpu" or cfg["threads"] != 4:
        raise ValueError("The controlled probe requires Ultralytics 8.3.230 and CPU four threads.")
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    # Native select_device otherwise resets the first prediction to its default eight threads.
    torch_utils.NUM_THREADS = 4
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch


def _network(cfg: Mapping[str, Any], model_cfg: Mapping[str, Any], adapter: Any) -> tuple[Any, Any]:
    import numpy as np
    import torch
    from ultralytics import YOLO

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if (
        model_cfg["name"] != "yolo11n_seg"
        or sha256_file(model_cfg["checkpoint"]) != model_cfg["sha256"]
    ):
        raise ValueError("This probe uses only the pinned original YOLO11n checkpoint.")
    wrapper = YOLO(str(model_cfg["checkpoint"]), task="segment")
    network = adapter.configure_yolo_network(
        wrapper, dict(cfg) | {"freeze_backbone_blocks": cfg["freeze_yolo_blocks"]}
    ).cpu()
    return wrapper, network


def _parameter_groups(network: Any) -> dict[str, list[str]]:
    import torch

    parameters = dict(network.named_parameters())
    bn = [
        f"{name}.{parameter}"
        for name, module in network.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for parameter, _ in module.named_parameters(recurse=False)
    ]
    head = f"model.{len(network.model) - 1}."
    return {
        "overall": list(parameters),
        "bn_affine": bn,
        "head": [name for name in parameters if name.startswith(head)],
        "trainable": [name for name, value in parameters.items() if value.requires_grad],
    }


def run_one_epoch(
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    output_dir: Path,
    adapter: Any,
    baseline_parameters: Mapping[str, Any] | None = None,
    *,
    evaluator: Any = None,
) -> dict[str, Any]:
    """Run the legacy recipe once; an injected adapter permits a separate mask-only arm.

    cfg['output'] remains the original pilot directory containing the frozen data.
    output_dir is a new diagnosis artifact directory, never the training data path.
    This function reports raw epoch-one results even when initialization is better.
    """
    from animal_eye_metrology.data.training_data import FrozenCocoDataset

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch = _runtime(cfg, output_dir)
    if cfg["epochs"] != 1 or cfg["batch_size"] != 2:
        raise ValueError("The controlled probe requires one epoch and microbatch size two.")
    fitting_path, dev_path = (
        cfg["output"] / "data/fitting.json",
        cfg["output"] / "data/development.json",
    )
    guard_data_paths(fitting_path, dev_path)
    evaluator = evaluator or training.evaluate
    wrapper, network = _network(cfg, model_cfg, adapter)
    parameters = dict(network.named_parameters())
    initial = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    if baseline_parameters is not None and (
        baseline_parameters.keys() != initial.keys()
        or any(not torch.equal(baseline_parameters[name], value) for name, value in initial.items())
    ):
        raise ValueError("Candidate initialization differs from the common original checkpoint.")
    groups = _parameter_groups(network)
    optimizer = torch.optim.SGD(
        [value for value in parameters.values() if value.requires_grad],
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        nesterov=False,
    )
    fitting = FrozenCocoDataset(fitting_path)
    if len(fitting) != 256:
        raise ValueError("Expected the unchanged 256-image fitting set.")
    source = json.loads(fitting_path.read_text())
    order = list(range(len(fitting)))
    random.Random(cfg["seed"]).shuffle(order)
    started = time.perf_counter()
    losses, norms, image_ids, vanished = [], [], [], []
    with (output_dir / "steps.jsonl").open("x") as log:
        for offset in range(0, len(order), 2):
            examples = [fitting[index] for index in order[offset : offset + 2]]
            images, targets = [row[0] for row in examples], [row[1] for row in examples]
            optimizer.zero_grad(set_to_none=True)
            adapter.set_yolo_train_mode(network)
            batch = adapter.make_yolo_batch(
                images, targets, network.names, source["categories"], cfg["image_size"], "cpu"
            )
            vector, components = network(batch)
            loss = vector.sum() / len(images)  # Deliberately retain the v1 recipe in every LR arm.
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite diagnosis loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [value for value in parameters.values() if value.requires_grad],
                cfg["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            ids = [int(row["image_id"]) for row in targets]
            scalar, norm_value = float(loss.detach()), float(norm.detach())
            losses.append(scalar)
            norms.append(norm_value)
            image_ids.extend(ids)
            for row in batch["letterbox"]:
                vanished.extend(row["empty_after_mask_downsampling_annotation_ids"])
            record = {
                "optimizer_step": len(losses),
                "image_ids": ids,
                "loss": scalar,
                "loss_components": components.detach().tolist(),
                "gradient_norm_before_clip": norm_value,
                "gradient_clipped": norm_value > cfg["gradient_clip_norm"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if len(losses) % 32 == 0:
                print(
                    json.dumps(
                        {"event": "lr_probe_progress", "candidate": output_dir.name, **record}
                    ),
                    flush=True,
                )
    train_seconds = time.perf_counter() - started
    if len(losses) != 128 or Counter(image_ids) != Counter({key: 1 for key in fitting.image_ids}):
        raise ValueError("Diagnosis did not update exactly once per fitting microbatch.")
    drift = relative_parameter_drift(initial, parameters, groups)
    metrics = evaluator(
        network, wrapper, model_cfg, dev_path, output_dir / "development_epoch_1", cfg, "cpu"
    )
    torch.save(
        {"model_state": network.state_dict(), "epoch": 1, "optimizer_steps": 128},
        output_dir / "trained_state.pt",
    )
    result = {
        "status": "complete",
        "candidate": output_dir.name,
        "model": "yolo11n_seg",
        "learning_rate": cfg["learning_rate"],
        "optimizer_steps": len(losses),
        "image_occurrences": len(image_ids),
        "unique_fitting_images": len(set(image_ids)),
        "development_mask_ap": metrics["coco_segm"]["mask_ap"],
        "mean_loss": sum(losses) / len(losses),
        "gradient_clip_fraction": sum(value > cfg["gradient_clip_norm"] for value in norms)
        / len(norms),
        "gradient_norm_min": min(norms),
        "gradient_norm_max": max(norms),
        "parameter_relative_l2_drift": drift,
        "training_seconds": train_seconds,
        "mask_downsampled_empty_annotation_ids": sorted(set(vanished)),
        "mask_sampling_policy": getattr(
            adapter, "MASK_SAMPLING_POLICY", "Provided adapter; see caller provenance."
        ),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "seed": cfg["seed"],
        "holdout_ground_truth_or_predictions_read": False,
        "optimizer_recipe": {
            "name": "SGD",
            "momentum": cfg["momentum"],
            "weight_decay_all_trainable_parameters": cfg["weight_decay"],
            "nesterov": False,
            "accumulation_steps": 1,
            "loss_divided_by_microbatch": True,
            "warmup": False,
            "ema": False,
            "all_batchnorm_eval": True,
            "frozen_blocks": "0..9",
        },
    }
    _json(output_dir / "result.json", result)
    return result


def _snapshot_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_lr_diagnosis(config_path: Path, output_directory: Path) -> dict[str, Any]:
    """Compare three LR arms against initialization using fitting/development only."""
    output = Path(output_directory).resolve()
    if not output.exists():
        output.mkdir(parents=True)
    if set(path.name for path in output.iterdir()) - {"provenance"}:
        raise FileExistsError("Refusing to overwrite an existing LR diagnosis.")
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    manifest_path = provenance / "snapshot_manifest.json"
    if not manifest_path.exists():
        from animal_eye_metrology.data import training_data
        from animal_eye_metrology.models import segmentation, training_yolo

        files = [
            Path(module.__file__)
            for module in (training, training_data, training_yolo, segmentation)
        ]
        files += [
            Path(config_path).resolve(),
            training.read_settings(Path(config_path))["model_config"],
        ]
        root = Path(__file__).resolve().parents[3]
        snapshot = {}
        for path in files:
            shutil.copyfile(path, provenance / path.name)
            snapshot[str(path.relative_to(root))] = sha256_file(path)
        _json(manifest_path, snapshot)
    snapshot = json.loads(manifest_path.read_text())
    for relative, digest in snapshot.items():
        if sha256_file(provenance / Path(relative).name) != digest:
            raise ValueError("Frozen diagnosis source snapshot changed.")
    cfg = training.read_settings(Path(config_path))
    base = cfg | {"epochs": 1, "device": "cpu", "evaluation_device": "cpu", "threads": 4}
    candidates = {name: base | {"learning_rate": rate} for name, rate in LR_CANDIDATES.items()}
    guard_lr_only(candidates)
    _runtime(base, output)
    adapter = _snapshot_module(provenance / "training_yolo.py", "aem_frozen_lr_adapter")
    frozen_training = _snapshot_module(provenance / "training.py", "aem_frozen_lr_evaluation")
    model_cfg = next(row for row in cfg["models"] if row["name"] == "yolo11n_seg")
    wrapper, network = _network(base, model_cfg, adapter)
    initial = {name: value.detach().cpu().clone() for name, value in network.named_parameters()}
    fitting_path, dev_path = (
        cfg["output"] / "data/fitting.json",
        cfg["output"] / "data/development.json",
    )
    guard_data_paths(fitting_path, dev_path)
    inputs = {
        str(path): sha256_file(path)
        for path in (
            fitting_path,
            dev_path,
            cfg["source"],
            cfg["model_config"],
            model_cfg["checkpoint"],
        )
    }
    baseline = frozen_training.evaluate(
        network, wrapper, model_cfg, dev_path, output / "untrained_development", base, "cpu"
    )
    results = {}
    for name, candidate in candidates.items():
        results[name] = run_one_epoch(
            candidate,
            model_cfg,
            output / name,
            adapter,
            initial,
            evaluator=frozen_training.evaluate,
        )
    after = {path: sha256_file(Path(path)) for path in inputs}
    if inputs != after:
        raise ValueError("The fixed data/config/checkpoint inputs changed during diagnosis.")
    best_name = max(results, key=lambda name: results[name]["development_mask_ap"])
    baseline_ap, best_ap = (
        baseline["coco_segm"]["mask_ap"],
        results[best_name]["development_mask_ap"],
    )
    root = Path(__file__).resolve().parents[3]
    workspace_changes = {
        relative: sha256_file(root / relative) if (root / relative).exists() else None
        for relative, digest in snapshot.items()
        if not (root / relative).exists() or sha256_file(root / relative) != digest
    }
    shutil.copyfile(Path(__file__), provenance / "training_diagnosis.py")
    report = {
        "status": "complete",
        "model": "yolo11n_seg",
        "untrained_development_mask_ap": baseline_ap,
        "candidate_results": results,
        "highest_development_training_candidate": best_name,
        "best_training_candidate_gain_vs_initialization": best_ap - baseline_ap,
        "selection_including_initialization": best_name
        if best_ap > baseline_ap + 1e-12
        else "untrained_baseline",
        "subsequent_training_started": False,
        "holdout_ground_truth_or_predictions_read": False,
        "only_changed_training_setting": "learning_rate",
        "source_snapshot_sha256": snapshot,
        "loaded_adapter_source_sha256": sha256_file(provenance / "training_yolo.py"),
        "diagnosis_source_sha256": sha256_file(Path(__file__)),
        "evaluation_source_file": inspect.getsourcefile(frozen_training.evaluate),
        "workspace_changes_since_snapshot": workspace_changes,
        "fixed_inputs_sha256": inputs,
        "fixed_inputs_unchanged": True,
        "limitations": [
            "One seed and one epoch on fixed development images; "
            "no causal claim beyond this controlled LR contrast.",
            "Initialization is an explicit candidate; smaller LR is not presumed superior.",
            "No holdout or official COCO val inference is performed.",
        ],
    }
    _json(output / "summary.json", report)
    print(
        json.dumps(
            {
                "event": "lr_diagnosis_complete",
                "baseline_ap": baseline_ap,
                "candidate_aps": {
                    name: row["development_mask_ap"] for name, row in results.items()
                },
                "selected": report["selection_including_initialization"],
            }
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_lr_diagnosis(args.config, args.output)


if __name__ == "__main__":
    main()
