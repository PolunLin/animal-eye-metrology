# Phase 2 — Animal Instance Segmentation

## 1. Role and Context

You are a Senior Computer Vision Engineer and Research Engineer working on the `animal-eye-metrology` project.

Follow the engineering standards and collaboration guidelines defined in `AGENTS.md`.

This project is an AI-based visual metrology system designed to perform:

1. Animal instance segmentation.
2. Animal eye localization and segmentation.
3. Intra-animal eye distance measurement.
4. Inter-animal right-eye distance measurement.
5. Quantitative evaluation and measurement validation.

Phase 1 has been completed and approved.

Your responsibility is to implement **Phase 2: Animal Instance Segmentation**.

The implementation must be scientifically defensible, reproducible, modular, and easy to understand.

Prioritize correctness and simplicity over architectural complexity.

---

## 2. Phase 2 Objective

Implement a reproducible animal instance segmentation pipeline using a pretrained segmentation model.

The primary research question is:

**Can a pretrained instance segmentation model reliably identify and segment individual animals in selected multi-animal COCO images?**

The pipeline must:

- Detect individual animal instances.
- Predict instance segmentation masks.
- Preserve animal category information.
- Map predictions to the original image coordinate system.
- Compare predictions against official COCO ground-truth annotations.
- Quantitatively evaluate segmentation performance.
- Visualize predictions and failure cases.

Do not implement eye localization, eye segmentation, or distance measurement in this phase.

---

## 3. Model Selection

### Primary Model

Use:

**YOLO11n-seg — Official COCO Pretrained Weights**

Framework: Ultralytics YOLO.

The model is selected for the initial proof of concept because:

1. It supports object detection and instance segmentation.
2. It provides a straightforward Python inference interface.
3. It supports COCO object categories.
4. It is suitable for integration into a larger computer vision pipeline.
5. Its relatively lightweight architecture reduces initial implementation complexity.

This selection is based on engineering feasibility, not an assumption that YOLO11n-seg provides the highest segmentation accuracy.

### Baseline Model

Reserve the following model for a future comparison:

**Mask R-CNN ResNet-50 FPN v2**

Do not implement the baseline model during the initial POC.

### Model Requirements

- Use official pretrained weights.
- Record the model name and version.
- Record the checkpoint source.
- Calculate and record the checkpoint SHA-256 hash.
- Document relevant licensing considerations.
- Do not train or fine-tune the model.

---

## 4. Dataset

Use the existing COCO 2017 Validation Dataset prepared during Phase 1.

### Input Dataset

Use the existing:

- COCO validation images.
- Official COCO instance annotations.
- Phase 1 image selection results.
- Phase 1 measurement feasibility results.

### POC Scope

Run inference on all **15 selected images**.

Among these images, 13 have been identified as preliminary measurement candidates.

These 13 images are intended for demonstration and technical feasibility analysis.

Do not treat them as an unbiased evaluation dataset.

### Data Integrity

- Do not modify official COCO annotations.
- Do not modify Phase 1 image selections.
- Do not modify manual eye visibility reviews.
- Preserve original COCO image IDs.
- Preserve original COCO category IDs.
- Preserve annotation IDs when referencing ground truth.

Predicted instance IDs must be handled separately from COCO ground-truth annotation IDs.

---

## 5. Instance Segmentation Pipeline

Implement the following pipeline:

```text
COCO Image
    |
    v
YOLO11n-seg Inference
    |
    v
Animal Category Filtering
    |
    v
Instance Segmentation Results
    |
    +-- Category
    +-- Confidence Score
    +-- Bounding Box
    +-- Instance Mask
    |
    v
Original Image Coordinate Mapping
    |
    v
COCO Ground Truth Comparison
    |
    v
Evaluation Metrics
    |
    v
Visualization and Reports
```

### Functional Requirements

For each input image:

1. Load the original image.
2. Run YOLO11n-seg inference.
3. Filter predictions to COCO animal categories.
4. Extract each detected animal instance.
5. Preserve the predicted category and confidence score.
6. Extract bounding boxes and segmentation masks.
7. Ensure all outputs use the original image coordinate system.
8. Associate predictions with their source image ID.
9. Save results in a reproducible format.

Do not assume that the order of predicted instances matches the order of COCO ground-truth annotations.

---

## 6. Ground Truth Matching

Use official COCO instance segmentation annotations as the reference.

Predicted instances must be matched against ground-truth instances using segmentation IoU.

### Matching Requirements

- Match predictions and ground truth within the same image.
- Require matching COCO categories.
- Use mask IoU for segmentation comparison.
- Prevent incorrect one-to-many matching.
- Record unmatched predictions.
- Record unmatched ground-truth instances.

For official COCO AP and AR evaluation, use the matching behavior implemented by COCOeval, including confidence ordering, IoU thresholds, and crowd/ignore handling.

For diagnostic instance-level reports, use a clearly documented matching rule and threshold.

Do not present a custom diagnostic matching procedure as equivalent to official COCOeval.

---

## 7. Evaluation Metrics

Evaluate segmentation performance using the following metrics.

### 7.1 Intersection over Union (IoU)

\[
IoU = \frac{|M_{pred} \cap M_{gt}|}{|M_{pred} \cup M_{gt}|}
\]

### 7.2 Dice Coefficient

\[
Dice = \frac{2|M_{pred} \cap M_{gt}|}{|M_{pred}| + |M_{gt}|}
\]

### 7.3 Precision

\[
Precision = \frac{TP}{TP + FP}
\]

### 7.4 Recall

\[
Recall = \frac{TP}{TP + FN}
\]

### 7.5 COCO Segmentation Metrics

Use official COCOeval to calculate:

- Mask AP@[0.50:0.95]
- Mask AP50
- Mask AP75
- Average Recall (AR), where applicable.

Clearly document the evaluation thresholds, category filtering, and evaluated image set.

### Evaluation Requirements

1. Use official COCO ground-truth masks.
2. Use official COCO category IDs.
3. Preserve COCO crowd and ignore semantics.
4. Avoid implementing a separate custom AP evaluator.
5. Clearly distinguish detection failures from mask quality failures.
6. Handle empty predictions correctly.
7. Report undefined metrics as unavailable rather than inventing values.

Do not claim generalization performance based solely on the 15 manually selected images.

---

## 8. Visualization

Generate visualizations for each selected image.

Each visualization should include:

- Original image.
- Ground-truth animal masks.
- Predicted animal masks.
- Predicted bounding boxes.
- Animal category labels.
- Confidence scores.

Use consistent visual conventions to distinguish predictions from ground truth.

Where practical, display matched and unmatched instances.

The goal is to make segmentation quality and failure cases easy to inspect.

Reuse the existing visualization utilities whenever possible.

Do not create a new visualization framework.

---

## 9. Execution Environment

Primary development environment:

**macOS Apple Silicon**

### Execution Strategy

1. Verify that the model loads successfully.
2. Run an initial inference test on CPU.
3. Validate the complete pipeline on a small number of images.
4. Run the full 15-image POC.
5. Optionally test MPS acceleration after CPU inference is working.

### Environment Requirements

- Use the existing Python project and `uv` environment.
- Avoid unnecessary dependency upgrades.
- Record Python, PyTorch, Ultralytics, and relevant package versions.
- Record the inference device.
- Record the input image size.
- Record confidence and IoU/NMS thresholds.
- Record the model checkpoint hash.

Do not assume MPS acceleration is supported without testing.

If MPS produces unsupported operations or inconsistent behavior, retain CPU as the reference execution environment.

---

## 10. Output Artifacts

Save all Phase 2 outputs under:

`outputs/phase2/`

Required artifacts:

### 1. `predictions_coco.json`

Predictions formatted for COCO segmentation evaluation.

Include:

- image_id
- category_id
- segmentation
- score

Use COCO-compatible segmentation encoding.

### 2. `instance_matches.csv`

Diagnostic instance matching results.

Include sufficient information to identify:

- Image ID.
- Predicted instance.
- Ground-truth annotation ID.
- Category.
- Match status.
- Mask IoU.
- Dice score.

### 3. `metrics.json`

Quantitative evaluation results.

### 4. `run_manifest.json`

Reproducibility metadata, including:

- Dataset identification.
- Dataset annotation hash.
- Selected image IDs.
- Model name and version.
- Checkpoint source and hash.
- Software versions.
- Execution device.
- Inference configuration.
- Execution timestamp.

### 5. Segmentation Overlays

Save visualization outputs for the selected images.

---

## 11. Engineering Constraints

Follow KISS and YAGNI principles.

Reuse the existing project structure.

Do not introduce unnecessary abstraction layers.

Do not:

- Refactor unrelated Phase 1 modules.
- Modify Phase 1 review data.
- Train or fine-tune models.
- Implement eye detection.
- Implement eye segmentation.
- Implement eye distance measurement.
- Create a new API.
- Create a database.
- Add Docker infrastructure.
- Create a new Jupyter Notebook.
- Implement additional segmentation models.
- Automatically commit or push changes.

Only introduce new dependencies when required.

---

## 12. Testing and Validation

Implement focused tests for critical functionality:

1. COCO animal category filtering.
2. Prediction coordinate consistency.
3. COCO prediction serialization.
4. Mask IoU and Dice calculations.
5. Prediction-to-ground-truth matching.
6. Empty prediction handling.

Avoid excessive testing infrastructure.

Run the existing test suite to ensure Phase 1 remains functional.

Record actual test results.

Do not fabricate test outcomes.

---

## 13. Acceptance Criteria

Phase 2 is complete when:

1. YOLO11n-seg loads successfully.
2. Inference runs on all 15 selected images.
3. Animal instances are correctly represented in the output.
4. Predicted masks are mapped to original image coordinates.
5. COCO-compatible predictions are exported.
6. Ground-truth matching is implemented and documented.
7. Segmentation evaluation metrics are generated.
8. Visualization outputs are available.
9. Failure cases are identified.
10. The pipeline can be reproduced using documented commands.
11. Existing Phase 1 functionality remains unaffected.

If any requirement cannot be completed, explicitly report the limitation rather than silently skipping it.

---

## 14. Completion Report

After implementation, provide a report in **Traditional Chinese** containing:

1. Implementation summary.
2. Files created or modified.
3. Model and checkpoint information.
4. Execution environment.
5. Inference results for the 15 images.
6. Quantitative evaluation metrics.
7. Representative success and failure cases.
8. Inference latency.
9. Test results.
10. Known limitations.
11. Recommended next steps.

Clearly distinguish verified experimental results from assumptions or expectations.

Do not proceed to Phase 3 automatically.

Do not commit or push changes without explicit approval.

Stop after completing Phase 2 and wait for my review.