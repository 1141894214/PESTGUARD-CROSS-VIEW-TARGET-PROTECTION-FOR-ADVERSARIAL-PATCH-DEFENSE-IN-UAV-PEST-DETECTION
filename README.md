# PestGuard: Target-Preserving Style Removal for Robust UAV Pest Detection

This repository provides the official implementation of **PestGuard**, a training-free image-preprocessing defense designed to improve the robustness of UAV-based forest pest detection against localized adversarial patch attacks.

PestGuard detects suspicious localized perturbations through style-removal responses while protecting stable target evidence through cross-view consistency. The complete defense pipeline integrates style-removal residual localization, robust median absolute deviation (MAD) calibration, cross-view target protection, adaptive mask generation, morphological mask refinement, and local canopy-context filling.

The defense operates entirely at the image level. The object detector and VGG-19 feature extractor remain frozen throughout inference, and no additional defense training is required.

The repository provides:

1. The complete PestGuard defense-inference pipeline.
2. Faster R-CNN, YOLO, and DETR detector interfaces.
3. Style-removal residual localization using frozen VGG-19 features.
4. MAD-based robust residual calibration.
5. Cross-view detection matching and target-protection-map construction.
6. Adaptive suspicious-region masking and morphological refinement.
7. Local canopy-context filling.
8. Patch construction and adversarial-attack utilities.
9. AP50 and Attack Success Rate evaluation utilities.
10. Intermediate visualization and debugging outputs.
11. Unit tests covering the major components of the proposed method.

---

## 1. Method Overview

PestGuard consists of the following stages.

### 1.1 Style-Removal Residual Localization

Given an input UAV image, PestGuard first adds a 10-pixel random border and performs a single style-removal update using a frozen pretrained VGG-19 network.

Style representations are constructed from the first five convolutional blocks of VGG-19 using Gram matrices. A fixed random image is used as the style reference.

Only the input image is updated during this process. All VGG-19 parameters remain frozen.

After one gradient-ascent step on the style loss, the updated image is clipped to the valid image range `[0,1]`.

The random border is subsequently removed, and the pixel-wise residual is calculated as

```text
r(u) = mean_RGB(abs(crop(z1)(u) - x(u)))
```

where:

* `x` is the original image,
* `z1` is the image after the one-step style update,
* `crop(z1)` removes the added border,
* `u` denotes a spatial image location.

The style-updated image is used only for residual computation and is not directly passed to the final object detector.

---

### 1.2 MAD-Based Robust Calibration

The residual response is normalized using the median absolute deviation (MAD):

```text
a(u) = (r(u) - median(r)) /
       (1.4826 * median(abs(r - median(r))) + eps)
```

The signed residual form is retained.

Therefore, pixels whose residual responses are below the global residual median can receive negative normalized scores.

This robust normalization suppresses the influence of globally distributed response variations while highlighting localized abnormal regions.

---

### 1.3 Cross-View Target Protection

PestGuard constructs a weakly transformed view of the input image using resize-pad transformation and mild photometric perturbation.

The same frozen object detector processes:

1. the original image; and
2. the weakly transformed view.

Predicted boxes from the weak view are mapped back to the coordinate system of the original image.

Class-wise Hungarian matching is then performed using IoU as the matching criterion.

Only matched box pairs satisfying

```text
IoU >= rho
```

are retained.

For each retained matched pair, PestGuard constructs a target-protection map:

```text
C(u) = max_j sqrt(p_j * p'_j)
       * 1[u in b_j intersect b'_j]
```

where:

* `b_j` and `b'_j` are matched detection boxes,
* `p_j` and `p'_j` are their corresponding confidence scores,
* `C(u)` represents stable target evidence at location `u`.

Pixels not covered by any retained box intersection receive

```text
C(u) = 0
```

The protection map reduces unnecessary masking inside regions that remain stable across image views.

---

### 1.4 Adaptive Mask Generation

The preliminary suspicious-region mask is generated using both the MAD-calibrated residual score and the cross-view protection map:

```text
M0(u) = 1[a(u) > tau0 + lambda * C(u)]
```

where:

* `a(u)` is the MAD-calibrated residual score,
* `tau0` is the base residual threshold,
* `lambda` controls the contribution of target protection,
* `C(u)` is the cross-view protection score.

A higher target-protection score increases the effective masking threshold and therefore reduces accidental removal of stable object evidence.

---

### 1.5 Mask Refinement

The preliminary mask is refined through three operations:

1. Removal of small connected components.
2. Morphological closing.
3. Morphological dilation.

Eight-connectivity is used for connected-component analysis.

The released configuration uses `3 × 3` structuring elements for morphological closing and dilation.

The resulting binary mask represents the final suspicious regions selected for local restoration.

---

### 1.6 Local Canopy-Context Filling

Each connected masked component is filled independently.

For a masked region, PestGuard constructs a surrounding unmasked annulus and collects valid neighboring pixels.

The channel-wise median of these surrounding pixels is then used to fill the selected region.

Pixels outside the final mask remain unchanged.

The resulting defended image is therefore obtained through localized modification of suspicious areas while preserving the original image elsewhere.

---

### 1.7 Final Detection

The locally restored image is passed to the same frozen object detector used before defense.

No detector weights are updated during PestGuard inference.

The overall processing flow is

```text
Input Image
    |
    +--> Style-Removal Update
    |        |
    |        +--> Residual Map
    |                 |
    |                 +--> MAD Calibration
    |
    +--> Original-View Detection
    |
    +--> Weak-View Detection
             |
             +--> Cross-View Matching
                      |
                      +--> Target-Protection Map
                               |
MAD Score ---------------------+
                               |
                               v
                      Adaptive Mask
                               |
                      Mask Refinement
                               |
                      Local Filling
                               |
                               v
                       Defended Image
                               |
                               v
                       Frozen Detector
```

---

# 2. Main Components

The current implementation includes the following components.

### Style-removal branch

* 10-pixel random border.
* Fixed random style-reference image.
* Frozen pretrained VGG-19.
* Features from the first five convolutional blocks.
* Gram-matrix style representation.
* One-step style-loss gradient ascent.
* Image clipping to `[0,1]`.
* Pixel-wise RGB-averaged residual computation.

### Robust calibration

* Residual median computation.
* Median absolute deviation.
* Signed MAD score.
* Numerical stabilization using `eps`.

### Cross-view protection

* Weak resize-pad transformation.
* Mild brightness and contrast perturbation.
* Original-view detector inference.
* Weak-view detector inference.
* Inverse box-coordinate mapping.
* Class-aware Hungarian matching.
* IoU-based matched-pair filtering.
* Confidence-weighted target-protection maps.

### Adaptive masking

* Protection-aware residual thresholding.
* Small-component filtering.
* Eight-connected component processing.
* Morphological closing.
* Morphological dilation.

### Local restoration

* Component-wise processing.
* Surrounding annulus construction.
* Exclusion of masked pixels from the filling source.
* Channel-wise median canopy-context filling.

### Detector interfaces

The repository includes interfaces for:

* Faster R-CNN;
* Ultralytics YOLO;
* HuggingFace DETR.

### Evaluation

The repository includes utilities for:

* AP50 evaluation;
* hiding-attack success evaluation;
* creation-attack success evaluation;
* attack-specific result aggregation;
* runtime recording.

### Adversarial patch utilities

The repository includes utilities for:

* patch-support construction;
* patch placement;
* local patch replacement;
* M-PGD perturbation projection;
* preservation of pixels outside the permitted patch region.

---

# 3. Experimental Configuration

The main PestGuard configuration is stored in

```text
config_proposed.json
```

The released configuration uses the following settings.

| Parameter                        |                           Value |
| -------------------------------- | ------------------------------: |
| Random border width              |                           10 px |
| VGG style features               | First five convolutional blocks |
| Style update step size           |                             1.0 |
| MAD numerical constant `eps`     |                          `1e-6` |
| Matching threshold `rho`         |                             0.5 |
| Detection confidence threshold   |                            0.25 |
| Base mask threshold `tau0`       |                             3.0 |
| Protection coefficient `lambda`  |                             2.0 |
| Minimum connected-component size |                           16 px |
| Connectivity                     |                  8-connectivity |
| Closing kernel                   |                           3 × 3 |
| Dilation kernel                  |                           3 × 3 |
| Annulus inner radius             |                            3 px |
| Annulus outer radius             |                            9 px |
| Weak-view scale range            |                       0.95–1.05 |
| Weak-view brightness range       |                       0.95–1.05 |
| Weak-view contrast range         |                       0.95–1.05 |
| Patch-to-box area ratio          |                             1.0 |
| M-PGD local `L_inf` bound        |                          16/255 |

Unless an experiment explicitly changes a parameter, the same PestGuard configuration is retained throughout the corresponding evaluation.

---

# 4. VGG-19 Style Features

PestGuard uses a frozen pretrained VGG-19 network for style-response extraction.

Style representations are constructed from the first five convolutional blocks.

For a selected feature tensor

```text
F_l
```

at layer `l`, the corresponding Gram matrix is constructed from the feature activations.

The style loss is computed as the sum of squared Frobenius distances between Gram matrices of the current image and the fixed style-reference image across the selected feature levels.

The VGG-19 network remains frozen throughout defense inference.

Only image pixels participate in the one-step style-removal update.

The repository accepts a local VGG-19 checkpoint through the command-line interface:

```text
--vgg-checkpoint /path/to/vgg19.pt
```

---

# 5. Installation

Install the required Python packages using

```bash
python -m pip install -r requirements.txt
```

GPU execution is recommended for experiments involving VGG-19 gradients and high-resolution UAV imagery.

For YOLO-based experiments, install Ultralytics:

```bash
python -m pip install ultralytics
```

For DETR-based experiments, install HuggingFace Transformers:

```bash
python -m pip install transformers
```

The detector and VGG-19 checkpoints used for evaluation should be provided locally.

---

# 6. Running PestGuard

## 6.1 Faster R-CNN

For torchvision Faster R-CNN, `--num-classes` includes the background class.

For example, a detector containing one foreground category normally uses:

```text
--num-classes 2
```

Example:

```bash
python run_pestguard.py \
  --input /path/to/test/images \
  --output /path/to/results \
  --detector fasterrcnn \
  --detector-checkpoint /path/to/fasterrcnn.pt \
  --num-classes 2 \
  --vgg-checkpoint /path/to/vgg19.pt \
  --config config_proposed.json \
  --device cuda:0
```

---

## 6.2 YOLO

Example:

```bash
python run_pestguard.py \
  --input /path/to/image.jpg \
  --output /path/to/results \
  --detector yolo \
  --detector-checkpoint /path/to/best.pt \
  --vgg-checkpoint /path/to/vgg19.pt \
  --config config_proposed.json \
  --device cuda:0
```

The YOLO interface supports local Ultralytics detector checkpoints.

---

## 6.3 DETR

Example:

```bash
python run_pestguard.py \
  --input /path/to/image.jpg \
  --output /path/to/results \
  --detector detr \
  --detector-checkpoint /path/to/local_detr_directory \
  --vgg-checkpoint /path/to/vgg19.pt \
  --config config_proposed.json \
  --device cuda:0
```

For DETR, `--detector-checkpoint` points to a local HuggingFace model directory containing the model parameters and corresponding image-processor configuration.

---

# 7. Output Files

For each processed input image, PestGuard saves the following outputs.

| File                     | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `defended.png`           | Final PestGuard-processed image                          |
| `style_updated_crop.png` | Image after the one-step style update and border removal |
| `weak_view.png`          | Weakly transformed view used for cross-view consistency  |
| `residual_preview.png`   | Visualization of the style-removal residual              |
| `mad_preview.png`        | Visualization of the MAD-calibrated residual score       |
| `protection_preview.png` | Visualization of the target-protection map               |
| `mask_preview.png`       | Visualization of the final refined mask                  |
| `maps.npz`               | Raw residual, MAD, protection, and mask arrays           |
| `result.json`            | Detection predictions and runtime information            |

The preview images are intended for visualization.

The numerical maps used by the algorithm are stored in

```text
maps.npz
```

and preserve the original floating-point values.

---

# 8. Detection Outputs

The output file

```text
result.json
```

records detection information from different stages of the pipeline.

These include:

```text
detections_original
```

for detections on the original image,

```text
detections_weak
```

for detections obtained from the weak view,

and

```text
detections_final
```

for detections obtained after PestGuard processing.

Runtime information is also recorded for subsequent efficiency analysis.

---

# 9. AP50 Evaluation

The repository provides

```text
evaluate_outputs.py
```

for evaluating saved detection outputs against YOLO-format annotations.

The evaluator uses:

* IoU threshold: `0.5`;
* 101-point interpolated AP;
* confidence floor: `0.001`;
* maximum predictions per image: `300`.

---

## 9.1 Annotation Requirements

For each evaluation image, a corresponding annotation file is required.

For an image containing no annotated target, the associated YOLO annotation file should be an empty `.txt` file.

For example:

```text
images/
    image_001.jpg
    image_002.jpg

labels/
    image_001.txt
    image_002.txt
```

A missing annotation file is treated as an evaluation error.

---

# 10. Faster R-CNN Evaluation Example

For torchvision Faster R-CNN, foreground labels commonly begin at `1`, while YOLO annotations commonly begin at `0`.

For a one-class detection task, use:

```bash
python evaluate_outputs.py \
  --results /path/to/results \
  --labels /path/to/test/labels \
  --class-ids 1 \
  --label-class-offset 1 \
  --prediction-key detections_final \
  --output /path/to/ap50.json
```

Here,

```text
--label-class-offset 1
```

maps YOLO class `0` to Faster R-CNN foreground class `1`.

---

# 11. YOLO Evaluation Example

For a zero-based YOLO detector:

```bash
python evaluate_outputs.py \
  --results /path/to/results \
  --labels /path/to/test/labels \
  --class-ids 0 \
  --label-class-offset 0 \
  --prediction-key detections_final \
  --output /path/to/ap50.json
```

---

# 12. Attack Success Evaluation

PestGuard evaluates adversarial attack success under two attack objectives.

## 12.1 Hiding Attack

A hiding attack attempts to suppress an existing pest-damaged-tree detection.

The corresponding utility evaluates whether the attacked target is successfully removed according to the matching and confidence criteria.

The implementation provides:

```text
hiding_success
```

for computing hiding-attack outcomes.

---

## 12.2 Creation Attack

A creation attack attempts to introduce a false pest-damaged-tree detection.

The implementation provides:

```text
creation_success
```

for computing creation-attack outcomes.

---

## 12.3 Attack Success Rate

In this repository,

**ASR denotes Attack Success Rate**.

Attack success is calculated separately according to the objective of each attack.

Hiding and creation attacks therefore use their corresponding success criteria during evaluation.

---

# 13. Adversarial Patch Utilities

The module

```text
pestguard/patches.py
```

contains the patch-processing functions used by the adversarial evaluation pipeline.

The available operations include:

1. construction of a patch-support region from a selected target bounding box;
2. patch placement inside the permitted spatial support;
3. patch replacement and update;
4. M-PGD perturbation projection;
5. local perturbation-bound enforcement;
6. preservation of all pixels outside the permitted patch region.

---

## 13.1 Patch Area

The standard configuration uses one patch placed on a selected crown.

The patch-to-target-box area ratio is

```text
1.0
```

for the corresponding attack protocol.

---

## 13.2 M-PGD Constraint

M-PGD restricts the perturbation to the local patch-support region.

The local perturbation satisfies

```text
||delta||_inf <= 16/255
```

inside the permitted patch region.

Pixels outside this region remain unchanged.

---

# 14. Adaptive BPDA+EOT Evaluation

Adaptive evaluation uses BPDA together with EOT.

A straight-through gradient estimator is used through the non-differentiable preprocessing operations during adaptive attack optimization.

The detector and VGG-19 model remain frozen during the defense procedure.

Adaptive evaluation therefore attacks the image-side defense pipeline without modifying the model parameters.

---

# 15. Weak-View Transformation

The cross-view branch constructs a weakly transformed image to identify stable detection evidence.

The released configuration uses:

```text
Scale:      0.95–1.05
Brightness: 0.95–1.05
Contrast:   0.95–1.05
```

The transformed image is resized and padded as required by the implementation.

Predictions from the weak view are mapped back to the coordinate system of the original image before cross-view matching.

---

# 16. Hungarian Matching

Detection matching is performed separately for each object class.

For original-view predictions

```text
B = {b_1, ..., b_n}
```

and mapped weak-view predictions

```text
B' = {b'_1, ..., b'_m},
```

the matching cost is derived from pairwise IoU.

Hungarian matching determines the assignment between the two prediction sets.

Pairs satisfying

```text
IoU >= 0.5
```

are retained by the standard configuration.

Only these retained pairs contribute to the target-protection map.

---

# 17. Target-Protection Map
