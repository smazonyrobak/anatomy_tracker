# 089 joint pose/map feedback: frozen synthetic-development result

## Decision

**Reject 089 as the selected-plane improvement and do not deploy it.** The joint pathway did train and the best candidate in the fixed 14-branch beam improved slightly, but the model did not learn to choose that candidate. On the same 246 eligible frozen 061 synthetic DEV sections, 085's selected 96-grid surviving-tissue CCF error was 2.587 mm and 089 batch 6,000 was **2.592 mm**. The best of those same 14 candidates improved from 0.992 to 0.961 mm. The prespecified ≥0.3 mm selected-error improvement gate therefore failed. Neither result is real-animal validation or a calibrated electrode-location probability.

The 085 lineage audit established that its feedback pose and spatial-map output weights were exactly zero. Unlike 085/088, 089 unfreezes both and backpropagates the final mapped physical error through a 083 atlas-comparison → pose/map → fit-summary → repeated pose/map path. The training trace confirms nonzero feedback-pose and feedback-map gradients. This directly tests the proposed coupling, but only a small correction was learned; it does not rescue selection.

## Frozen checkpoint comparison

Eight synthetic-deformation-plan-equal means, exact 086 8-old/6-anchor branch IDs per section, 1,024 fixed surviving pixels per section. All errors below are 3-D CCF distances unless stated otherwise.

| 089 batch | Fitted-score-selected mapped error | Best-of-14 mapped error | Prior-selected mapped error | Score vs lower mapped error, within-section rho | Fitted selected regret |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 = 085 | 2.587 mm | 0.992 mm | 2.635 mm | 0.140 | 1.595 mm |
| 1,000 | 2.632 mm | 0.978 mm | 2.610 mm | 0.154 | 1.654 mm |
| 3,000 | 2.635 mm | 0.968 mm | 2.611 mm | 0.155 | 1.667 mm |
| 6,000 | **2.592 mm** | **0.961 mm** | **2.596 mm** | **0.141** | **1.631 mm** |

At batch 6,000, the fitted-selected branch's initial rigid error was 2.634 mm and after the two feedback passes was 2.616 mm; its mean centre shift was 92 µm. The selected local-map magnitude was 95 µm, up from 52 µm at batch zero. The best-of-14 initial rigid error changed from 1.022 to 0.995 mm; its corrected rigid error was 0.990 mm. These tiny physical effects cannot account for the 1.63 mm selection regret. The updated prior's beam retained a mean 12.54 of the 14 frozen branch IDs, so the fixed-beam readout isolates correction but is not a complete measurement of the changed-prior inference beam.

Appearance did not improve consistently. At batch 6,000 versus zero, fitted selection changed from 2.424 to **2.562 mm** for exact-black exteriors (worse), 2.714 to 2.630 mm for raw backgrounds, and 2.510 to 2.430 mm for imperfect-brush inputs. These are separately drawn geometries, not paired background-only counterfactuals. The selected five-canvas-point pose error was 6.90 mm at zero and 7.12 mm at batch 6,000, but this extrapolates into often non-tissue canvas corners and the selected branch can change; surviving-pixel CCF distance is the decision metric here. The within-batch first/last-ten logged training means fell from 2.13 to 1.86 mm, whereas DEV selection did not; training improvement alone was not accepted.

## Integrity and limits

The trainer exited normally after 6,000 eligible fresh synthetic TRAIN sections in 6,589 attempts, with distinct physical-section IDs and all 64 TRAIN deformation bases. No real labels, external pretrained weights, public DeepSlice benchmark or final-test animals were used. It saved frozen checkpoints at 0/1,000/3,000/6,000. Its completed receipt records config SHA-256 `15c501c7276321ef25fd865e893d3428935b248807e7b2ac45fed8a04b519e33`, training log `d2712f0b6656166b51f9c3500fa5326412968c364674d214e00195fddd3413a1`, and final checkpoint `dd49c68416f406d9d9a75f54b235f9c0984cadf0aea0b462febb87e6fbd11f0a`. The preregistered evaluator verified these, exact source/parent/panel/086 hashes, draw provenance and step-zero candidate-level parity with 085. Its own receipt records 984 section rows and 13,776 candidate rows, with candidate SHA-256 `e3d1d032e519a1bd8c29fb4bf56c5a37fc3ec316da8cae716874f8b992e802c8`, section `44db305d4ec17f0942007c161a733f6ce557a1ddd54af11219597d3eafce21c1`, and summary `cbcb1c4cea261c81814b79cd3143a981335cd2eadd90a60c4484721f28b85fdd`. An independent raw-row audit rehashed all four evaluator outputs and recomputed every section's selected/best branch, regret, unchanged beam, and eight-plan-equal selected/best means (float32 rounding tolerance 0.001 µm): pass. No image was reopened for this decision. All development and results remained on I:.

## Next targeted work

The dominant failure is **candidate discrimination**, not insufficient local warp. Do not extend the same 089 run or tune a scalar score blend on this DEV panel. Investigate why the 083 local cost volume/085 fitted scorer barely ranks physically good arbitrary planes, then train candidate-level anatomical compatibility with explicit hard negatives and a listwise physical target, while preserving the trained joint pose/map path. A matched frozen-beam DEV readout must show a material selected-error gain and no black-background regression before increasing training exposure or using real-animal validation. The 089 training yielded only ~92 µm average centre revision, so if a stronger selector still cannot use the candidate bank, revisit the pose-update representation or correspondence search range rather than hiding the gap with a larger deformation field. Retain untouched animal-level tests, future calibration, GUI delivery and fair public DeepSlice comparison as later gates.
