# 046 spatial matching: useful ranking, ineffective local offset

The spatial fine matcher continued the scratch-lineage 045 encoder and trained a newly initialized pair/correction head for 2,000 batches on 4,000 fresh arbitrary-plane synthetic TRAIN sections. Frozen 019/041 supplied actual off-plane top-eight proposals and top-32 keys. Training finished in 571 seconds at 3.81 GB peak GPU memory. The predeclared five-checkpoint comparison on all visible queries of the unchanged 177-section, eight-identity synthetic DEV panel finished in 406 seconds. Its independent read-only verifier passed the training lineage, source/checkpoint/row hashes, split separation, exact batch-zero 045 reproduction, and all summaries. Raw data and checkpoints are frozen at `I:/AnatomyTracker/runs/pose_fine_subgrid_046_pilot` and `I:/AnatomyTracker/runs/pose_fine_subgrid_046_development_eval`.

Equal-identity selected-point recall and mean CCF error for the *physically best of the original eight 019 proposals*:

| 046 batch | Raw selected key ≤0.5 mm | Corrected point ≤0.5 mm | Corrected point ≤1.5 mm | Corrected mean error |
| ---: | ---: | ---: | ---: | ---: |
| 0 (045 continuation) | 8.1% | 8.1% | 53.0% | 1.765 mm |
| 500 | 10.6% | 10.9% | 70.3% | 1.335 mm |
| 1,000 | 11.8% | 11.8% | 74.6% | 1.255 mm |
| 1,500 | 11.5% | 11.7% | 73.9% | 1.262 mm |
| 2,000 | 12.6% | **12.9%** | **76.4%** | **1.216 mm** |

The prior top-one branch also improves from 33.0% to 45.1% within 1.5 mm, but its corrected mean point error remains 2.632 mm. All three appearance strata improve in the best-eight ≤0.5-mm score. The predeclared **25% within 0.5 mm** stage criterion fails at every checkpoint; none is promoted to pose fitting or GUI use. “Best eight” is an oracle choice based on known truth for diagnostic purposes, not model selection.

Most of the gain comes from **ranking the coarse keys**, not the continuous correction. At batch 2,000, raw best-eight recall within 1.5 mm is 76.2% versus 76.4% corrected, and mean error falls only 1.224→1.216 mm. Across all query cells, the mean absolute change in point-error magnitude is about 32 µm (maximum about 121 µm). The trained offset cannot bridge the grid's ≳0.5-mm spacing. A single pair of global descriptors plus 5×5 feature correlation did not yield a useful 3D residual, especially through-plane; this is project-specific evidence, not a claim that local refinement is impossible.

Keep the 046 scoring checkpoint as an experimental correspondence initializer, not a qualified model. The next principled change should **render and compare several nearby 3D atlas positions around a shortlisted key** so sub-grid displacement has direct image evidence, then pool spatially coherent matches for full pose and deformation fitting. A fitted candidate-quality signal must overturn the 019 prior when appropriate, and the fitting loss must train the coordinate predictor. More batches of the same offset head are not the immediate priority. The repeatedly used synthetic DEV identities are not independent biological validation; no uncertainty is calibrated and no DeepSlice comparison is warranted.
