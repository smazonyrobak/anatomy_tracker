# v3 input-appearance census result, 2026-10-09

The frozen [TRAIN-only protocol](SLIDE_V3_APPEARANCE_CENSUS_001_PROTOCOL_20261009.md) completed on 256 eligible, independently drawn physical planes from 288 attempts. All 256 physical IDs were distinct. Independent post-run SHA-256 checks matched the receipt for rows, attempts, and summary. The sampler changed input-mixture probabilities only; it did not train or alter a model checkpoint.

| Whole-canvas metric | Real donor-equal TRAIN (512) | v2 synthetic stress (256) | v3 synthetic routine (256) |
| --- | ---: | ---: | ---: |
| Raw/no-brush count | 512 | 100 | 171 |
| Raw with selected exact-black exterior | 512 observed dark-border images (not code-labeled) | not separately audited | 131 |
| Brush-outline available | 0 | 156 | 85 |
| Mean exact-zero image fraction | 0.695 | 0.571 | 0.633 |
| Mean fraction below 0.03 | 0.790 | 0.650 | 0.705 |
| Mean outer-border fraction above 0.03 | 0.000 | 0.141 | 0.077 |

The preregistered descriptive QC expectations were mixed: raw with selected exact-black exterior was **131/256 = 51.2%** (required ≥45%); outline availability was **85/256 = 33.2%** (required ≤35%); outer-border fraction was **0.0768** (expected <0.07), so that last expectation **failed narrowly**. No threshold was moved and no resampling was done. A post-hoc decomposition shows that the retained 17 gray-background raw cases had mean bright-border fraction 0.88, while the 131 raw exact-black and 23 raw near-black cases each averaged about 0.02. Thus the nonzero aggregate border is mainly the deliberately retained non-black deployment variability, not a mistaken claim that it matches the Allen corpus. Even black cases can have tissue at the edge under some independent frame draws. Real images are stored at 192², synthetic at 256², and the sources differ; exact-zero fractions are not acquisition calibration.

The v3 event counts were fragment 45, fold 56, bubble 39, and seam 56, down from v2's 129/138/104/154 in separate 256-plane censuses. These are independently sampled cases with co-occurring events; neither rate is a measured laboratory prevalence. The physical appearance of these events, stain variation, scanner behavior, and donor variation still require representative acquired-slide comparison. The inherited model checkpoint has **not** seen v3 and this result is not evidence of improved alignment. The v2 stress set remains useful for challenge evaluation. No sealed holdout, public benchmark, or numerical uncertainty calibration was accessed.

Output: `I:/AnatomyTracker/runs/slide_v3_train_appearance_census_001`; rows SHA-256 `5a2913549df1c600545d5c200f6b326f6a72f65d57cd25830cf4418410a5e0e6`, attempts `596cd0010920fc2f451872d4ea878e7269265e21a9743100b8772f775c493c35`, summary `b73f6a7955f2ffca87a145e474a8d6c677c79b4404081a30e8e8cc672552b482`.
