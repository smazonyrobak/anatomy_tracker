# AtlasPose development and release protocol

AtlasPose predicts `[AP from bregma (um, anterior positive), L-R tilt (degrees), D-V tilt (degrees)]` from a coronal mouse-brain section. The auxiliary orientation logit resolves the 180-degree ambiguity left after smart-mask roll canonicalization; it is not another anatomical coordinate.

## Deployed v6: historical evidence only

The historical local ConvNeXt-Tiny v6 model was selected on synthetic Allen CCFv3 sections and trained on 100,000 unique synthetic views. Its sealed synthetic-test MAE was 58.72 um AP, 0.934 degrees L-R, and 1.052 degrees D-V.

The 148-section published DeepSlice set was subsequently used as development feedback. It is therefore **not** an untouched holdout and cannot support a production-release claim. On that set, v6 reached 245.20 um, 1.639 degrees, and 3.996 degrees MAE, versus 174.26 um, 1.463 degrees, and 1.268 degrees for the published DeepSlice outputs. The optional 20% AtlasPose / 80% DeepSlice vote reached 140.61 um, 1.20 degrees, and 1.63 degrees. These results explain the current GUI default—DeepSlice—but must not be reused as final evidence for v7.

The v6 model is an experimental predictor, not a demonstrated DeepSlice replacement. It is not source-approved for a new release.

## v7 candidates

The pending v7 experiment compares the same task-specific pose heads and registered/synthetic data across three ImageNet-initialized backbones:

1. `legacy_xception.tf_in1k` — the closest backbone control to DeepSlice. [DeepSlice](https://www.nature.com/articles/s41467-023-41645-4), [Xception](https://arxiv.org/abs/1610.02357)
2. `convnext_tiny.fb_in22k_ft_in1k` — a modern convolutional backbone with Apache-2.0 pretrained weights. [ConvNeXt](https://arxiv.org/abs/2201.03545)
3. `maxvit_tiny_rw_224.sw_in1k` — a convolution/attention hybrid with Apache-2.0 pretrained weights. [MaxViT](https://arxiv.org/abs/2204.01697)

The non-commercial ConvNeXtV2 FCMAE checkpoint is deliberately excluded. Export provenance records the exact timm identifier, upstream URL or Hugging Face identifier, declared license, and a SHA-256 of the initialized backbone state.

Three pose representations are compared: direct physical regression, physical AP/tilt bins with residuals, and QuickNII OUV regression. All receive the same tolerance-normalized physical-pose objective. The coarse-anatomy decoder is auxiliary training supervision only.

## Data and augmentation

`synthetic_atlas.py` renders exact oblique planes from `average_template_25.nrrd`; `annotation_25.nrrd` provides the known brain support. Training covers AP +500 to -4500 um and L-R/D-V tilt -35 to +35 degrees. Smart-mask preprocessing removes input canvas scale and in-plane roll as nuisance variables.

Synthetic views include the requested range of nonlinear warps, arbitrary rotation, 0.5-1.5x scale, missing tissue and edge-to-edge occlusion, contrast/gamma/offset and polarity changes, non-black backgrounds, local exposure, repeating tile/vignette artifacts, blur/noise, bright specks and blowouts, and artificial tears. Clean and mild views remain present. Fixed seeded manifests make every comparison reproducible and prevent validation/test images from entering training.

Registered Allen serial two-photon sections supply real-image training and validation. Dataset manifests, downloaded images, quality exclusions, atlas files, code, dependencies, and pretrained initialization are hashed in the run/export provenance. The published DeepSlice development set is excluded from selection. The final registered holdout is sealed and may be opened once, only after the candidate is frozen.

## Prespecified experiment

- Screen direct, binned, and OUV heads at 20,000 unique views with three training seeds.
- Continue the two best heads to 100,000 views. The winning ConvNeXt-Tiny/head runs are reused rather than trained a second time during the backbone comparison.
- Compare Xception, ConvNeXt-Tiny, and MaxViT-Tiny at 100,000 views with the selected head and the same three seeds.
- Run 20,000-view renderer, consistency, and anatomy ablations against an explicit selected-model control.
- Train the selected configuration on up to 1,000,000 unique views with validation-based early stopping.
- Evaluate the registered test split after selection, then run the one-shot sealed DeepSlice comparison.

Model-family decisions use registered-validation animal-level hierarchical bootstraps with a prespecified tie order. The release gate requires AP MAE <=60 um, L-R MAE <=0.90 degrees, D-V MAE <=1.75 degrees, absolute AP bias <=25 um, AP 95th percentile <=150 um, worst 500-um AP-band MAE <=90 um, and worst product MAE <=90 um. It must also beat the published DeepSlice ensemble independently on AP, L-R, and D-V in the sealed paired animal-level comparison. No v7 performance is claimed until that gate passes.

Passing metrics alone does not deploy a model. The runtime accepts AtlasPose only when the ONNX model, metadata, and sealed release evidence exactly match hashes pinned in application source. Promotion emits proposed hashes for human review and never edits those source pins.

## Running v7

Use the CUDA environment and explicit workspace/data roots:

```powershell
$env:PYTHONPATH = "$PWD"
$env:ATLAS_POSE_V7_WORKSPACE = "J:\AtlasPoseTraining_v7"
$env:ATLAS_POSE_REGISTERED = "J:\AtlasPoseTraining_v7\allen_registered_full_20260811"
$env:ATLAS_POSE_ATLAS = "$PWD\data\Allen Brain Atlas 25um"
C:\Users\slic\miniconda3\envs\npixel_analysis\python.exe training\train_atlas_pose_v7.py
```

The selected FP32 ONNX binary is intentionally ignored by Git because it exceeds GitHub's normal 100 MB single-file limit. CPU and every available application accelerator provider must agree with PyTorch within the export tolerances before promotion.

DeepSlice's repository and PyPI metadata disagree about redistribution terms; obtain maintainer clarification before closed-source redistribution. AtlasPose uses Allen-derived data and pretrained timm weights; this project does not itself grant redistribution rights for Allen data or derived weights.
