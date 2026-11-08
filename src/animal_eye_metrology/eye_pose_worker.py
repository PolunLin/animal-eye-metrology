"""Official HRNet inference in an isolated environment, with no eye reference inputs."""

import argparse
import hashlib
import json
import logging
import random
import resource
import sys
import time
from importlib.metadata import distribution, distributions
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
CONFIG = (
    "configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_probe(image: np.ndarray, box: np.ndarray, flips: list[int]) -> dict[str, Any]:
    """Known QA coordinates exercise the actual official affine and flip transforms."""
    import cv2
    from mmpose.datasets.transforms import GetBBoxCenterScale, TopdownAffine
    from mmpose.structures.bbox import get_warp_matrix
    from mmpose.structures.keypoint import flip_keypoints

    x0, y0, x1, y1 = box[0]
    known = np.array([[[x0, y0], [x1, y1], [(x0 + x1) / 2, (y0 + y1) / 2]]])
    transformed = TopdownAffine(input_size=(256, 256))(
        GetBBoxCenterScale(padding=1.25)(
            {"img": image.copy(), "bbox": box.copy(), "keypoints": known.copy()}
        )
    )
    center, scale = transformed["input_center"], transformed["input_scale"]
    restored = transformed["transformed_keypoints"] / [256, 256] * scale + center - scale / 2
    error = float(np.max(np.abs(restored - known)))
    warp = get_warp_matrix(center, scale, rot=0, output_size=(256, 256))
    inverse = cv2.invertAffineTransform(warp)
    inverse_error = float(
        np.max(np.abs(cv2.transform(transformed["transformed_keypoints"], inverse) - known))
    )
    landmarks = np.arange(34, dtype=np.float32).reshape(1, 17, 2) + 10
    visible = np.ones((1, 17), dtype=np.float32)
    flipped, fv = flip_keypoints(landmarks.copy(), visible, (256, 256), flips)
    recovered, _ = flip_keypoints(flipped.copy(), fv, (256, 256), flips)
    first_expected = np.array([255 - landmarks[0, 1, 0], landmarks[0, 1, 1]])
    flip_error = float(np.max(np.abs(recovered - landmarks)))
    swap_ok = bool(np.array_equal(flipped[0, 0], first_expected))
    if max(error, inverse_error, flip_error) > 1e-3 or not swap_ok:
        raise ValueError("Official affine/flip QA failed.")
    return {
        "qa_not_research_ground_truth": True,
        "restore_error_px": error,
        "inverse_affine_error_px": inverse_error,
        "flip_roundtrip_error_px": flip_error,
        "left_channel_takes_mirrored_right": swap_ok,
        "tolerance_px": 1e-3,
        "input_center": center.tolist(),
        "input_scale": scale.tolist(),
        "original_to_input_affine": warp.tolist(),
        "input_to_original_affine": inverse.tolist(),
    }


def run(request_path: Path) -> None:
    import cv2
    import mmpose
    import torch
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import register_all_modules

    started = time.perf_counter()
    request = json.loads(request_path.read_text())
    allowed = {
        "checkpoint",
        "checkpoint_sha256",
        "output",
        "cases",
        "seed",
        "threads",
        "mmpose_commit",
        "compatibility_probe",
        "mmpose_source_manifest",
        "mmpose_source_manifest_sha256",
    }
    if set(request) != allowed or request["seed"] != 42 or request["threads"] != 4:
        raise ValueError("Unexpected request fields or execution policy.")
    root = request_path.parent
    output = (root / request["output"]).resolve()
    output.mkdir()
    checkpoint = (root / request["checkpoint"]).resolve()
    if digest(checkpoint) != request["checkpoint_sha256"]:
        raise ValueError("HRNet checkpoint digest mismatch.")
    package = distribution("mmpose")
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    module_root = Path(mmpose.__file__).parent
    source_path = root / request["mmpose_source_manifest"]
    if digest(source_path) != request["mmpose_source_manifest_sha256"]:
        raise ValueError("Official source manifest digest changed.")
    official = json.loads(source_path.read_text())
    if official["commit"] != request["mmpose_commit"]:
        raise ValueError("MMPose source does not match the approved commit.")
    for relative, expected in official["files_sha256"].items():
        if digest(module_root / relative) != expected:
            raise ValueError(f"Installed MMPose source differs from official commit: {relative}")
    config_path = module_root / ".mim" / CONFIG
    if not config_path.is_file():
        raise ValueError("Official packaged config not found.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    register_all_modules()
    t0 = time.perf_counter()
    model = init_model(
        str(config_path),
        str(checkpoint),
        device="cpu",
        cfg_options={"model.backbone.init_cfg": None, "model.test_cfg.output_heatmaps": True},
    )
    # Repeat the official PyTorch state-dict load strictly, invoking registered conversion hooks.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data["state_dict"]
    strict_result = model.load_state_dict(state, strict=True)
    del checkpoint_data, state
    names = model.dataset_meta["keypoint_id2name"]
    flips = list(model.dataset_meta["flip_indices"])
    if (
        model.dataset_meta["dataset_name"] != "ap10k"
        or [names[i] for i in range(3)] != ["L_Eye", "R_Eye", "Nose"]
        or len(names) != 17
        or flips[:3] != [1, 0, 2]
    ):
        raise ValueError("Incorrect animal keypoint metadata; no COCO fallback allowed.")
    if not model.test_cfg["flip_test"] or not model.test_cfg["shift_heatmap"]:
        raise ValueError("Official test-time settings changed.")
    load_seconds = time.perf_counter() - t0
    results = []
    for case in request["cases"]:
        if set(case) != {"image_id", "prediction_id", "image", "image_sha256", "bbox_xyxy"}:
            raise ValueError("Pose case must not contain references or manual prompts.")
        path = root / case["image"]
        if digest(path) != case["image_sha256"]:
            raise ValueError("Frozen image digest mismatch.")
        image = cv2.imread(str(path))  # official LoadImage convention: BGR
        if image is None:
            raise ValueError("Image decoding failed.")
        box = np.array([case["bbox_xyxy"]], dtype=np.float32)
        record = {"image_id": case["image_id"], "prediction_id": case["prediction_id"]}
        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                samples = inference_topdown(model, image, bboxes=box, bbox_format="xyxy")
            elapsed = time.perf_counter() - t0
            if len(samples) != 1:
                raise ValueError("Expected exactly one pose for one predicted animal box.")
            sample = samples[0]
            points = np.asarray(sample.pred_instances.keypoints)
            scores = np.asarray(sample.pred_instances.keypoint_scores)
            heatmaps = np.asarray(sample.pred_fields.heatmaps)
            if points.shape != (1, 17, 2) or scores.shape != (1, 17):
                raise ValueError("Unexpected keypoint shape.")
            if heatmaps.shape != (17, 64, 64) or not all(
                np.isfinite(a).all() for a in (points, scores, heatmaps)
            ):
                raise ValueError("Invalid heatmaps or keypoint values.")
            geo = geometry_probe(image, box, flips)
            if not np.allclose(sample.metainfo["input_scale"], geo["input_scale"]):
                raise ValueError("Probe and actual model preprocessing disagree.")
            # Independently decode saved native heatmaps and check official original-grid mapping.
            decoded, decoded_scores = model.head.decoder.decode(heatmaps)
            restored = (
                decoded
                / np.array(sample.metainfo["input_size"])
                * np.array(sample.metainfo["input_scale"])
                + np.array(sample.metainfo["input_center"])
                - np.array(sample.metainfo["input_scale"]) / 2
            )
            decode_error = float(np.max(np.abs(restored - points)))
            if decode_error > 1e-3 or not np.allclose(decoded_scores, scores):
                raise ValueError("Saved heatmaps do not reproduce output coordinates.")
            np.save(output / f"{case['prediction_id']}_heatmaps.npy", heatmaps, allow_pickle=False)
            record.update(
                status="passed",
                keypoints_xy=points[0].tolist(),
                scores=scores[0].tolist(),
                inference_seconds=elapsed,
                geometry=geo,
                decode_error_px=decode_error,
                visibility_status="unverified",
                input_size=list(sample.metainfo["input_size"]),
                failure_reason="",
            )
        except Exception as exc:
            LOGGER.exception("Pose failed for %s", case["prediction_id"])
            record.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
        results.append(record)
    manifest = {
        "status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
        "request_sha256": digest(request_path),
        "checkpoint_sha256": digest(checkpoint),
        "model_load_seconds": load_seconds,
        "results": results,
        "mmpose_commit": request["mmpose_commit"],
        "official_source_files_verified": len(official["files_sha256"]),
        "official_source_manifest_sha256": digest(source_path),
        "installation_direct_url": direct,
        "config_sha256": digest(config_path),
        "keypoint_names": [names[i] for i in range(17)],
        "flip_indices": flips,
        "strict_state_dict": str(strict_result),
        "device": "cpu",
        "dtype": "float32",
        "seed": 42,
        "threads": torch.get_num_threads(),
        "batch_size": 1,
        "flip_test": True,
        "shift_heatmap": True,
        "wall_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if sys.platform.startswith("linux") else 1),
        "environment": {d.metadata["Name"]: d.version for d in distributions()},
        "mmpose_source_sha256": {
            str(p.relative_to(module_root)): digest(p) for p in sorted(module_root.rglob("*.py"))
        },
    }
    (output / "pose_results.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    if request["compatibility_probe"] and manifest["status"] != "passed":
        raise ValueError("Compatibility stage failed; do not run Stage B.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    run(parser.parse_args().request.resolve())
