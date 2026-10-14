# 306: logit-only pose ranking did not meet its advancement gate

The [frozen protocol](BLIND_RANKING_ONLY_306_PROTOCOL_20261014.md) ran from commit `9a976b4` against the verified fluorescence/older-donor-rehearsed 305 checkpoint. It drew 3,200 independently sampled eligible synthetic TRAIN sections for 800 four-section optimizer batches, plus two distinct 64-section panels from synthetic bases 56–63. Only the mixture/reflection probability rows of three final pose layers were trainable. A separate post-exit audit matched all ten output hashes, source/protocol bindings, unique physical-section IDs, every per-base readout and the raw gate; it compared every parent and selected checkpoint tensor and confirmed all non-logit rows exactly unchanged. No atlas fit, deformation training, DEV/CAL/FINAL animal, external model or public benchmark was used.

| Batch | Fresh inner-panel first-choice rigid-site error | Top-16 ≤1.5-mm capture |
| ---: | ---: | ---: |
| 0, frozen parent | 2.715 mm | 58/64 |
| **200, selected** | **2.558 mm** | **58/64** |
| 400 | 2.685 mm | 58/64 |
| 800 | 2.622 mm | 58/64 |

The selected batch-200 gain was **0.157 mm**, below the predeclared **0.25-mm inner-panel minimum**; only four of eight base means improved. The runner therefore correctly did **not** inspect its separate confirmation panel or load acquired images for the real-data guard. The synthetic gate failed, and no 306 checkpoint is promoted. Top-16 capture staying at 58/64 while first-choice error changes confirms that this experiment altered ranking, not candidate geometry; it does not establish a usable improvement.

**Decision:** do not extend this unchanged probability head, loosen the gate or spend a longer run on scalar/listwise recalibration of the same frozen features. Alongside the failed 305 joint selector and the nine-section blind local-information result, this rules out three cheap selection fixes, not the full arbitrary-plane task. The next work should test a change to the information available to the global proposal/selection model—especially real-domain and synthetic-image formation or candidate-relative spatial features—using fresh TRAIN development sections and strict controls. Exact-pose local mapping, acquired donor-separated validation and calibrated electrode-region probabilities remain separate required gates. The current 305 checkpoint remains experimental and the GUI must not use 306.

Frozen run: `I:/AnatomyTracker/runs/blind_ranking_only_306_001`; independently audited completion receipt SHA-256 `87bd0e2e35fa2a2920c6e6e581166e717b4196dde0c8735f05c9fe5011457a23`. All model-development outputs remain on `I:`.
