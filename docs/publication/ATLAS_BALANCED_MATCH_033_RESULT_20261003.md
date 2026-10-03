# Balanced match and pose continuation 033 — frozen development result

The 032 selector was continued without optimizer or RNG reset for 6,000 more batches: 12,000 fresh synthetic TRAIN sections, 18,000 in the selector lineage overall. The loss explicitly separated match ranking, no-match detection and differentiable physical plane fitting. All 740 DEV rows across four checkpoints and every saved affine were independently replayed; checkpoint, draw, training, source and output hashes passed. The 185 DEV sections represent eight synthetic deformation identities, not eight independent animals. No real donor or public benchmark entered this stage.

| Total selector batches | Correct selected among available top-16 matches | False match among unavailable queries | Mean fitted rigid error |
| ---: | ---: | ---: | ---: |
| 3,000 parent | 0% | 0% | 3.585 mm |
| 5,000 | 0.62% | 0.66% | 3.355 mm |
| 7,000 | 1.81% | 1.53% | 3.188 mm |
| 9,000 | 1.08% | 1.01% | **3.100 mm** |

The preregistered gate (≥50% available-match recall, ≤20% unavailable false-match rate and <2.5 mm rigid error) **failed** at every continuation checkpoint. The point-estimate error improved by 0.486 mm from the 032 parent but remains far too large. A secondary, mathematically appropriate *grouped* decision chooses “some match” when the summed probability of the 16 matches exceeds the no-match probability, then picks the strongest match. It was added to the evaluation before that evaluation ran, but was not the preregistered gate. At batch 7,000 it recovered 12.90% of available matches with 28.76% false matches; at batch 9,000, 11.88% with 21.27% false matches. Even this decision does not make the model useful.

The continuation demonstrates modest pose learning but not reliable correspondence selection, and it has no anatomical fitting-to-pose feedback yet. More unchanged training on the same architecture/loss is not supported. The 034 frozen-pool audit separately shows that 031's truth-blind geometric rank contains a <1 mm seed for only 46.55% of sections even after 32 seeds; a single selected seed is <1 mm for 15.79%. Thus an expensive full-image fitter over a small top-ranked retrieval beam is unlikely by itself to solve capture. Keep the 025 cross-modal patch encoder as reusable evidence, but make the direct probabilistic pose model the main route and target a physically supervised atlas-conditioned update that actually moves its candidates. Do not infer calibration or GUI readiness from these synthetic results.
