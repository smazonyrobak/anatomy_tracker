# Fixed TRAIN-derived contrast control: helpful, insufficient

The single contrast arm declared in `SAGITTAL_CONTRAST_CONTROL_019_PROTOCOL_20261008.md` multiplied the once-inverted input by 4.2593648215 and clipped it, with checkpoint, donor-frozen 80-section weak-DEV set, geometry, channels, and branch decision otherwise unchanged. The baseline and control use the same sections. Output is `I:/AnatomyTracker/runs/sagittal_contrast_control_019_001`; `rows.jsonl` SHA-256 is `8389391438c09f03f8c4794d974ff005c0881f7cc2c57b8c823dbc508b0353a4` and `summary.json` SHA-256 is `996c3ba50f9d42adfdc8c94bec5155296decb24743c5de7b655f50a01c7866bf`.

| Donor-equal weak-Allen disagreement | Original | Fixed gain | Improvement |
| --- | ---: | ---: | ---: |
| Selected five-point CCF distance | 8.695 mm | **6.021 mm** | 2.674 mm |
| Selected antipodal normal angle | 63.77° | **52.35°** | 11.43° |
| Truth-best 16×2 branch distance, diagnostic only | 5.018 mm | 4.081 mm | 0.936 mm |

The prespecified major-contrast-contributor gate required **both** ≥2 mm position improvement and ≥20° normal improvement; it **fails** because normal improves only 11.43°. One donor's normal error actually worsened (44.1°→52.3°), while the other three improved. The fixed gain changes the model's behavior substantially, confirming sensitivity to an acquired-versus-synthetic intensity gap, but the remaining 6.0-mm/52° errors are unusable and cannot be rescued by a simple contrast multiplier. Independent read-back verified 80 unique section IDs, four donors, the selected branch against saved full 16×2 matrices, all donor-equal means and the output hash.

Do not select another gain on this weak-DEV cohort. The priority is additional physical sagittal TRAIN donors and more realistic acquired-appearance training with preserved full-sphere synthetic pose coverage; retain a new donor-disjoint sagittal reserve before making a later generalization claim. This is not evidence against full-angle geometry in principle, nor calibrated uncertainty, dense-warp accuracy, physical steep-oblique performance or GUI readiness.
