# 151 posthoc correspondence diagnostic: audited result

This is a **mechanism diagnostic, not a new qualification gate or blind result**. The prespecified 150 gate remains failed. The 151 source and [protocol](COARSE_ATLAS_POSE_151_POSTHOC_ORACLE_PROTOCOL_20261011.md) were committed before the run. It used the frozen 150 matcher and fresh synthetic DEV panel, with synthetic truth substituted only to measure ceilings. No weights were trained or selected on this panel.

All 243 eligible sections from eight new synthetic DEV deformation plans were evaluated at the exact true plane and at the truth-best nearby branch of the frozen 132 candidate beam: 486 unique rows. The 13 prespecified ineligible sections were the only exclusions. The nearby subset below contains 181 branches whose initial rigid error was ≤1.5 mm. Each value is the plan-equal mean rigid CCF error over all valid native-256 tissue pixels, after the **same fixed pose fit**:

| Nearby branch, 181 sections | Error (mm) |
| --- | ---: |
| Initial candidate | 0.841 |
| Learned correspondence and learned effective weights | 0.778 |
| True nearest reachable atlas key, learned weights | 0.607 |
| True nearest reachable atlas key, valid-point weights | 0.546 |
| True continuous observed-to-atlas target, valid-point weights | 0.540 |
| True rigid plane target, valid-point weights | 0.526 |

Every one of the 19,064 valid nearby query sites had a supported atlas key inside the candidate-local reach limit. The key lattice itself is therefore not the missing mechanism in this panel. The conditional gap from learned to true keys with learned weights is about 0.171 mm; changing to valid-point weights after supplying true keys saves another 0.061 mm. These are **conditional oracle substitutions, not additive causal effects**. With the exact true plane, the learned fit was 0.232 mm; true keys with learned weights were worse at 0.369 mm, while true keys with valid weights gave 0.095 mm, continuous observed-to-atlas targets 0.064 mm, and true rigid targets 0.0008 mm. That reversal warns against interpreting a single oracle substitution as an implementable gain.

The 0.526 mm residual from a nearby candidate even with true rigid correspondences points to a limitation in the current pose-fit update—possibly ridge strength, trust limits, or parameterization—rather than an intrinsic inability to find matching atlas keys. It does **not** isolate which fit component is responsible. A TRAIN-only ridge/trust ablation should establish that before the fit rule changes. Matching and reliability learning also need to improve on TRAIN sections with the same whole-box framing and box-uniform offsets as the failed fresh DEV panel. A new DEV panel, not this diagnosed panel, must judge the next model.

The independent read-only audit rehashed all five run/source files and three frozen input completions; recomputed every summary stratum from raw rows; verified 243 × 2 rows, finite fits, 100% key reachability, TRAIN/DEV identity separation, and no expert, final-animal, pretrained, or public-benchmark use. Frozen 151 completion SHA-256 is `fdd62676d9912359b89b9d5ad46ed6dbd3c0b255934ef7c1d1889daac6b595e4`; rows SHA-256 is `27be2ac4f10966283aa97575993785cb74042421d3ef4cfcbc982437e7b2b1dd`. The full raw results remain at `I:/AnatomyTracker/runs/coarse_atlas_pose_151_posthoc_oracle`.

This synthetic oracle does not establish all-angle accuracy on biological sections, a deployable joint pose/deformation model, calibrated uncertainty, GUI readiness, or performance against DeepSlice.
