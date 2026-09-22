# PestGuard Research Code (Runnable Skeleton)

This project implements the core defense-inference pipeline described in the PestGuard paper. It also refers to the [official AntiStyler demonstration repository](https://github.com/IdanYankelev/AntiStyler) for the construction of VGG-19 style features and Gram matrices. The implementation is independent and does not copy the AntiStyler notebook.

The current version is intended to validate the method pipeline, interface integration, and boundary conditions. Hyperparameters that are not explicitly specified in the paper are clearly labeled as implementation defaults. These defaults alone must not be used to claim reproduction of the AP, ASR, or ablation results reported in the paper.

## Method Overview

PestGuard is an image-preprocessing defense that requires no additional defense training. Its pipeline consists of the following steps:

1. Perform a one-step style-removal update to obtain a residual map that highlights regions potentially affected by adversarial patches.
2. Robustly calibrate the residual using the median absolute deviation (MAD).
3. Compare detections from the original image and a weakly transformed view to construct a cross-view target-protection map.
4. Generate an adaptive mask from the residual score and the protection map.
5. Refine the mask morphologically and fill the selected regions using surrounding canopy context.
6. Send the processed image to the same frozen detector for final inference.

Only the input image is updated during the defense process. The object detector and VGG-19 parameters remain frozen.

## Implemented Components

1. Add a 10-pixel random border around the original image and generate a fixed random style-reference image. VGG-19 remains frozen, and a single gradient-ascent step on the style loss is applied only to the input image. The updated image is clipped to `[0,1]`.
2. Construct Gram matrices and the style loss from features extracted from the first five convolutional blocks of VGG-19. The exact layer indices are explicitly defined in the code to avoid ambiguity between convolutional blocks and individual convolutional layers.
3. Remove the random border and compute the pixel-wise channel-averaged absolute residual:

   ```text
   r(u) = mean_RGB(abs(crop(z1)(u) - x(u)))
   ```

4. Compute the signed MAD-calibrated score according to the equation in the paper:

   ```text
   a(u) = (r(u) - median(r)) /
          (1.4826 * median(abs(r - median(r))) + eps)
   ```

   The numerator is not converted to an absolute value. Pixels below the residual median can therefore receive negative scores.
5. Run the detector on the original image and the weakly transformed view, then map the weak-view predictions back to the original image coordinates.
6. Perform class-wise Hungarian matching using IoU as the matching criterion and retain only pairs whose IoU is not lower than `rho`.
7. Construct the target-protection map only inside the intersection of each retained matched-box pair:

   ```text
   C(u) = max_j sqrt(p_j * p'_j) * 1[u in b_j intersect b'_j]
   ```

   If a pixel is not covered by the intersection of any retained matched pair, then `C(u) = 0`.
8. Generate the preliminary adaptive mask:

   ```text
   M0(u) = 1[a(u) > tau0 + lambda * C(u)]
   ```

9. Remove small connected components, apply morphological closing, and dilate the result to obtain the final mask. Eight-connectivity, 3×3 structuring elements, and the minimum component size are explicit defaults of the current implementation.
10. For each connected component in the final mask, compute the channel-wise median from the surrounding unmasked annulus and use it to fill the component.
11. Preserve the original pixels outside the mask and send only the locally filled original image to the same frozen detector for final inference. The style-updated image is used only to compute the residual and is not used directly as the final detector input.
12. Provide adapters for Faster R-CNN, Ultralytics YOLO, and HuggingFace DETR detectors.
13. Save defended images, weak views, style-updated images, visualization heatmaps, raw floating-point maps, and detection outputs.
14. Provide utility functions for 101-point interpolated AP50, hiding-attack success, and creation-attack success.
15. Provide utilities for constructing patch support regions, applying patch replacements, and enforcing the local `16/255` norm constraint for M-PGD.
16. Include unit tests for residual computation, MAD calibration, inverse box mapping, Hungarian matching, protection-map construction, mask processing, and local filling.

## Settings Explicitly Specified in the Paper

- The random border width is 10 pixels.
- A fixed random image is used as the style reference.
- A frozen, pretrained VGG-19 is used.
- Style features are extracted from its first five convolutional blocks.
- The style loss is the sum of squared Frobenius norms between Gram matrices at the selected layers.
- A single style-loss gradient-ascent step is performed, and image values are clipped to `[0,1]`.
- The weak view includes resize-pad and mild photometric jitter.
- Detection boxes of the same class are matched using Hungarian matching based on IoU.
- Mask refinement includes small-component removal, morphological closing, and dilation.
- Each masked region is filled using the channel-wise median of its surrounding annulus.
- The default attack places one patch on a selected crown with a patch-to-box area ratio of 1.0.
- M-PGD uses an `L_inf` bound of `16/255` inside the patch region.
- Adaptive evaluation uses BPDA+EOT with a straight-through gradient estimator.
- The detector and VGG-19 remain frozen during defense inference.

## Information Requiring Real Experimental Resources or Author Configuration

- Trained detector checkpoints and the class-ID mapping used by each dataset.
- The exact VGG-19 checkpoint. The paper only states that a frozen, pretrained VGG-19 is used; it does not specify the pretraining dataset, weight version, or whether task-specific fine-tuning was performed.
- The exact VGG-19 layer indices corresponding to the five convolutional blocks.
- The experimental values of `eta`, `beta`, or their product used as the effective style-update step size.
- The distribution, resolution handling, and random seed used to generate the random style-reference image.
- The input normalization applied before VGG-19 inference.
- The experimental values of `rho`, `tau0`, `lambda`, and `eps`.
- The minimum connected-component threshold, morphological structuring-element sizes, and iteration counts.
- The inner and outer radii of the local filling annulus and the fallback rule when too few valid annulus pixels are available.
- The exact scaling, brightness, contrast, and padding ranges used for weak-view generation.
- The exact training, validation, and test splits for PDT, FDLC, and PWD.
- Detector training configurations, best-checkpoint selection records, and detector-specific preprocessing.
- Complete objectives, transformation distributions, iteration counts, step sizes, and optimization logs for EOT, DPatch, T-SEA, and adaptive BPDA+EOT.
- Complete hiding- and creation-success protocols for multiple targets, multiple predictions, and no-target images.
- The exact evaluator configuration used to produce the AP50 and ASR values reported in the paper.

Without this information, the current code can execute and validate the algorithmic pipeline, but it cannot automatically reproduce the numerical results in the paper.

## Installation

```bash
python -m pip install -r requirements.txt
```

The code can run on CPU, but a GPU is recommended for the VGG-19 gradient step on 640×640 images.

Install the corresponding optional dependency when using YOLO or DETR:

```bash
python -m pip install ultralytics
python -m pip install transformers
```

Do not use randomly initialized detectors or VGG-19 models for paper experiments.

## Configuration

The example configuration file is:

```text
config_proposed.json
```

Values not explicitly specified in the paper are implementation defaults. Before formal evaluation, calibrate them on the validation set and save the final configuration, random seeds, and execution logs.

By default, the current implementation normalizes VGG inputs using the ImageNet mean and standard deviation and generates the random style reference from a fixed-seed uniform distribution over `[0,1]`. These are implementation choices and must not be presented as settings explicitly specified in the paper.

## Inference

### Faster R-CNN

`--num-classes` includes the background class. For example, a torchvision Faster R-CNN model with one foreground class normally uses `2`, with the foreground label usually set to `1`.

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

### YOLO

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

### DETR

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

For DETR, `--detector-checkpoint` must point to a local HuggingFace directory containing both the model weights and image-processor configuration. The adapter does not download weights automatically.

## Output Files

The program saves the following files for each input image:

| File | Description |
|---|---|
| `defended.png` | Final locally filled defense image |
| `style_updated_crop.png` | Image after one style update and border removal |
| `weak_view.png` | Weakly transformed image used for cross-view consistency |
| `residual_preview.png` | Display preview of the residual map |
| `mad_preview.png` | Display preview of the MAD-calibrated score |
| `protection_preview.png` | Display preview of the cross-view target-protection map |
| `mask_preview.png` | Display preview of the final mask |
| `maps.npz` | Raw floating-point residuals, MAD scores, protection maps, and masks |
| `result.json` | Detections and runtime information for the original, weak-view, and final defended images |

`maps.npz` contains the raw floating-point values used by the algorithm. Preview PNG files are normalized for visualization and must not be interpreted directly as algorithm scores.

## Evaluating PDT with YOLO-Format Annotations

`evaluate_outputs.py` reads saved predictions and YOLO-format `.txt` annotations.

- A no-target image must have a corresponding empty `.txt` file.
- A missing annotation file raises an error.
- If the detector is torchvision Faster R-CNN and the YOLO foreground class is `0`, use `--label-class-offset 1 --class-ids 1`.
- If the detector itself uses zero-based class IDs, normally use `--label-class-offset 0 --class-ids 0`.

Faster R-CNN example:

```bash
python evaluate_outputs.py \
  --results /path/to/results \
  --labels /path/to/test/labels \
  --class-ids 1 \
  --label-class-offset 1 \
  --prediction-key detections_final \
  --output /path/to/ap50.json
```

YOLO example:

```bash
python evaluate_outputs.py \
  --results /path/to/results \
  --labels /path/to/test/labels \
  --class-ids 0 \
  --label-class-offset 0 \
  --prediction-key detections_final \
  --output /path/to/ap50.json
```

The current evaluator uses 101-point interpolated AP50, a confidence floor of `0.001`, and at most 300 predictions per image. These are evaluator settings of the current implementation; the paper does not provide the corresponding details. Before comparing against the reported results, verify the original image list, class mapping, prediction-filtering rules, and AP implementation.

`hiding_success` and `creation_success` evaluate hiding and creation attacks separately. The code does not combine them into a single ASR without an explicitly defined aggregation protocol.

## Scope of the Attack Utilities

`pestguard/patches.py` implements the following basic operations:

- Construct a patch support region from a target box and a specified area ratio.
- Insert a patch into the permitted image region.
- Project M-PGD updates onto the local `16/255` norm constraint inside the patch region.
- Keep all pixels outside the patch region unchanged.

These utilities do not constitute complete implementations of EOT, DPatch, T-SEA, or BPDA+EOT. A complete attack additionally requires an objective function, initialization scheme, iteration count, step size, random transformation distribution, target-selection policy, and stopping rule.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use a lightweight mock detector to verify that:

- The full inference pipeline is connected correctly.
- Image coordinates and detection boxes are transformed and mapped back correctly.
- Hungarian matching respects class identities and the IoU threshold.
- The protection map responds only inside matched-box intersections.
- MAD scores retain their signed form.
- Morphological mask processing and local filling satisfy boundary conditions.
- Empty detections, empty masks, and connected components touching image boundaries are handled safely.

These tests validate program logic only and provide no evidence of experimental performance.

## Relationship to AntiStyler

AntiStyler uses style-removal responses to locate potential adversarial patches and masks suspicious regions. Building on this idea, PestGuard uses a one-step style update, robust MAD calibration, cross-view target protection, and local median filling to reduce the accidental removal of useful features from small pest-damaged crowns.

This project refers to the AntiStyler demonstration for VGG-19 style-feature and Gram-matrix construction. It does not reproduce the notebook line by line and does not automatically treat AntiStyler hyperparameters as PestGuard experimental settings.

## Parameter Sources and Status

| Parameter | Current Default | Source/Status |
|---|---:|---|
| Random border | 10 px | Explicitly specified in the paper |
| VGG style features | First five convolutional blocks | Explicitly specified in the paper; exact layer indices must be fixed in the implementation |
| `style_step_size` | 1.0 | Implementation default; not numerically specified in the paper and requires experimental calibration |
| `eps` | 1e-6 | Numerical-stability default; the paper only requires `eps > 0` |
| `rho` | 0.5 | Implementation default; not numerically specified in the paper |
| Detection confidence floor | 0.25 | Implementation default; not numerically specified in the paper |
| `tau0` | 3.0 | Implementation default; not numerically specified in the paper |
| `lambda` | 2.0 | Implementation default; not numerically specified in the paper |
| Minimum connected component | 16 px | Implementation default; not numerically specified in the paper |
| Connectivity | Eight-connectivity | Implementation default; not specified in the paper |
| Closing structuring element | 3×3 | Implementation default; not numerically specified in the paper |
| Dilation structuring element | 3×3 | Implementation default; not numerically specified in the paper |
| Annulus inner/outer radii | 3 px, 9 px | Implementation default; not numerically specified in the paper |
| Weak-view scale | 0.95–1.05 | Implementation default; the paper only specifies a mild transformation |
| Weak-view brightness | 0.95–1.05 | Implementation default; the paper only specifies mild photometric jitter |
| Weak-view contrast | 0.95–1.05 | Implementation default; the paper only specifies mild photometric jitter |
| M-PGD local bound | 16/255 | Explicitly specified in the paper |
| Patch-to-box area ratio | 1.0 | Explicitly specified in the paper |

## Reproducibility Recommendations

For each formal experiment, save the following information:

1. Dataset splits and the complete test-image list.
2. Hashes of the detector and VGG-19 checkpoints.
3. Class names and class-ID mappings.
4. The complete configuration file.
5. Python, PyTorch, torchvision, CUDA, and detector-framework versions.
6. All random seeds.
7. Complete configurations and optimization logs for every attack.
8. Raw per-image predictions, AP evaluator inputs, and ASR decisions.
9. Images before and after defense, together with raw floating-point intermediate maps.
10. Failure cases and runtime exception logs.

Strict comparison with the paper requires identical data, weights, attacks, evaluator settings, and method parameters.

## Disclaimer

The current project is a runnable research skeleton derived from the method description publicly available in the paper. It does not include unpublished checkpoints, dataset splits, attack logs, or private training configurations. The code can validate the method pipeline, but it does not guarantee reproduction of the reported numerical results without the original experimental resources.
