# 290: the fixed local correlation has a true-offset signal but poor peaks

The frozen no-training audit at `I:/AnatomyTracker/reports/spatial_pair_cost_290_002` completed. The first invocation (`..._001`) stopped before any readout on a reflection-dtype error; the repair at `30ecdc5` changed only the boolean cast and output path. Receipt SHA-256: `1b5c593cfb6c2c41d52ffc1f067b9130f2eeae078d4831dae469bd66f5d986d2`; config SHA-256: `362579b5ac593fd718cdc7a7e5cad0fae3798485602b14690cbb88d88e43900c`. Source, protocol and output hashes matched. The 34,367 raw site rows span 64 distinct frozen 289 TRAIN-inner sections; independent regrouping reproduces every reported mean and hit fraction. Each panel image and target hash was checked before scoring.

| Exact-pose, evaluable synthetic sites | Intact tissue | Displaced fragments |
| --- | ---: | ---: |
| Sites, evaluable / prespecified | 4,934 / 32,768 | 1,368 / 1,599 |
| True rounded offset scores above zero offset | 80.5% | 71.3% |
| Mean true-minus-zero correlation cost | +0.185 | +0.124 |
| Highest-scoring offset within 2 pixels of true | **35.5%** | **13.9%** |
| Zero offset within 2 pixels of true | 75.3% | 2.7% |
| Mean peak-offset error | 4.59 pixels | 6.98 pixels |
| Mean true through-plane residual | 0.023 mm | 0.025 mm |

The fixed-cost sufficiency criterion **fails**: it required 60% intact and 50% fragment peak hits, each at least ten percentage points better than zero. For intact tissue, a strong zero-shift prior actually beats the unconstrained correlation peak because most evaluable intact displacements are small. For fragments, the peak is better than zero but is still wrong within two pixels on 86% of evaluable sites. The positive true-vs-zero costs are a real directional cue, consistent with 288's global exact-coordinate oracle, but not an accurate local correspondence field. This explains why adding more capacity downstream of the unchanged 289 cost is a weak bet.

**Next change:** train image/atlas descriptors from random initialization using explicit true-offset supervision and a physically bounded displacement prior/spatial consistency within each tissue piece, then compare a new frozen panel. Preserve 3-D residual learning for the small through-plane component and allow discontinuities across detached fragments. A local match score must also eventually survive real-image contrast differences; this result only concerns Allen-derived synthetic tissue. It does not justify fit→pose feedback, calibrated probabilities, GUI promotion, or a DeepSlice comparison. Full-sphere cutting support remains the intended model domain; this diagnostic does not demonstrate that arbitrary physical sections are impossible.
