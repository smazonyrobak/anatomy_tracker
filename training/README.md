# Atlas-pose CNN development

This pipeline compares three ImageNet-initialized CNN regressors on the same one-pass stream of synthetic Allen CCFv3 coronal sections. The primary output is:

`[AP from bregma (µm, anterior positive), L–R tilt (degrees), D–V tilt (degrees)]`

The exported model also produces an auxiliary orientation logit. It resolves the 180° ambiguity left after mask-PCA roll canonicalization and is used only to construct the correct slice-to-atlas affine; it is not a fourth anatomical coordinate.

## Architectures

1. **Xception** is the task-specific control. DeepSlice uses an ImageNet-pretrained Xception backbone, two 256-unit ReLU layers, and nine linear QuickNII-coordinate outputs. This implementation uses the same backbone/head pattern with three standardized pose outputs. DeepSlice was trained on 131k slide-mounted sections, 443k serial two-photon sections, and about 0.9M synthetic sections, so a synthetic-only model must pass an untouched real-histology benchmark before it can be described as a replacement. Sources: [DeepSlice paper](https://www.nature.com/articles/s41467-023-41645-4), [DeepSlice source](https://github.com/PolarBean/DeepSlice/blob/main/DeepSlice/neural_network/neural_network.py), [Xception paper](https://arxiv.org/abs/1610.02357).
2. **EfficientNetV2-S** tests a similarly sized backbone designed for fast accelerator training with fused MBConv blocks. Source: [EfficientNetV2 paper and official implementation](https://arxiv.org/abs/2104.00298).
3. **ConvNeXt-Tiny** tests a modern conventional-convolution backbone with strong classification, detection, and segmentation transfer results. Sources: [ConvNeXt paper](https://arxiv.org/abs/2201.03545), [official implementation](https://github.com/facebookresearch/ConvNeXt).

U-Net was considered because it is strongly supported for biomedical segmentation, but pixel segmentation is not the requested global pose target. The GUI smart-brush mask already supplies the relevant foreground support and scale information directly. Source: [U-Net paper](https://arxiv.org/abs/1505.04597).

## Synthetic data

`synthetic_atlas.py` renders oblique planes directly from `average_template_25.nrrd` and uses `annotation_25.nrrd` as the exact training outline. Rendering and geometric transforms run on CUDA. The raw section is independently rotated and scaled by 0.5–1.5×; its known outline then supplies the same roll and scale canonicalization available from the GUI smart brush, preventing pixel size or canvas orientation from leaking the target.

Per-image probabilities match the requested design:

- 90% receive one to three distinct optical defects: contrast/tone changes, selective intensity-band exposure, and/or repeating square-tile vignettes. When contrast modification is selected, half of those cases also invert polarity.
- 60% receive smooth nonlinear deformation. The deformation audit found positive Jacobian determinants across 2,000 checked samples, so the augmentation distorts without folding the coordinate field over itself.
- 100% receive a random full-circle in-plane rotation and independent 0.5–1.5× scaling.
- 40% receive occlusion: 4% of all images use an edge-to-edge blackout and 36% use a random polygon, usually cortex-anchored and capped at 50% of brain area.

The 5k/10k/15k/20k/30k sets are nested prefixes of one persisted 100k training manifest, so a model sees each generated training image only once within a run. AP uses uniform Latin-hypercube-like coverage across +500 to −4500 µm in every split. Training, validation, and sealed test data use separate seeded manifests. Validation drives architecture and checkpoint selection; the sealed test split is evaluated only after model selection.

## Controlled model comparison

At the 30k stage, validation MAE was:

| Architecture | AP MAE (µm) | L–R MAE (°) | D–V MAE (°) | Normalized score |
|---|---:|---:|---:|---:|
| Xception | 126.1 | 2.01 | 3.36 | 1.4825 |
| EfficientNetV2-S | 106.5 | 1.75 | 2.73 | 1.2303 |
| ConvNeXt-Tiny | **94.4** | 1.92 | **2.09** | **1.0560** |

ConvNeXt-Tiny beat EfficientNetV2-S by 0.174 normalized-score units in a paired 10,000-resample bootstrap; the 95% interval was −0.195 to −0.154, and every resample favored ConvNeXt. It was therefore selected for the 100k run.

## Final v6 evidence

The selected ConvNeXt-Tiny was trained on the complete 100k unique-image manifest with validation-based checkpoint selection and early-stopping monitoring. Validation continued improving through the scheduled run, so early stopping did not terminate before 100k. The sealed synthetic test result was:

| Metric | AP (µm) | L–R (°) | D–V (°) |
|---|---:|---:|---:|
| MAE | 58.72 | 0.934 | 1.052 |
| 95th-percentile absolute error | 149.77 | 2.557 | 2.849 |

Auxiliary 180° orientation accuracy on the sealed synthetic test was 98.8%. CPU and DirectML ONNX export checks agreed within 0.0025 µm for AP and within 0.00003° for either tilt; orientation logits agreed within 0.000005.

The untouched real-histology benchmark contains 148 published DeepSlice sections inside the requested +500 to −4500 µm AP domain:

| Predictor | AP MAE (µm) | L–R MAE (°) | D–V MAE (°) |
|---|---:|---:|---:|
| Published DeepSlice outputs | 174.26 | 1.463 | **1.268** |
| AtlasPose v6 | 245.20 | 1.639 | 3.996 |
| 80% DeepSlice + 20% AtlasPose | **140.61** | **1.20** | 1.63 |

AtlasPose's corresponding 95th-percentile errors were 515.32 µm, 4.554°, and 9.858°. It is therefore an independent experimental predictor, not a demonstrated DeepSlice replacement. DeepSlice remains the GUI default. Weighted voting is opt-in and defaults to 20% AtlasPose because that weight minimized the benchmark aggregate; although the vote improved AP and L–R MAE in this set, its D–V MAE remained worse than DeepSlice alone. Every result still requires overlay review.

## Running

Use the existing CUDA environment:

```powershell
$env:PYTHONPATH = "$PWD\training"
$env:ATLAS_POSE_WORKSPACE = "G:\AtlasPoseTraining"
C:\Users\slic\miniconda3\envs\npixel_analysis\python.exe training\train_atlas_pose.py
```

Generated manifests, checkpoints, predictions, and diagnostic plots stay in `G:\AtlasPoseTraining`. The selected deployable model is FP32 ONNX. Its approximately 112 MB binary is deliberately ignored by Git because it exceeds GitHub's normal 100 MB single-file limit. `models/AtlasPose/atlas_pose.json` records its SHA-256, output contract, and validation metadata. A local production build requires both files; `TrajectoryTracker.spec` bundles them into `models/AtlasPose` beside the executable.

The bundled DeepSlice models have contradictory repository/PyPI license metadata; obtain maintainer clarification before closed-source redistribution. AtlasPose was trained from Allen atlas-derived synthetic sections. This project does not itself grant redistribution or commercial-use clearance for the Allen source data or derived weights, so review the applicable Allen Institute terms before shipping either.
