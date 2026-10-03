# 045 off-plane patch matching: partial gain, not a pose solution

The scratch-lineage 025 patch encoder was continued for 500 batches using 1,000 fresh independent arbitrary-plane synthetic TRAIN sections. The original 019 probabilistic pose predictor and 041 3D shortlist generator stayed frozen. Actual shortlisted atlas keys—not an oracle plane—were rendered as oriented, finite-thickness 64×64 patches. The run used 2.59 GB peak GPU memory and finished in 136 seconds. The predeclared matched batch-zero/batch-500 comparison on 177 eligible synthetic DEV sections took 154 seconds and was independently audited: **PASS**. Raw training, draw provenance, checkpoints and evaluation rows are at `I:/AnatomyTracker/runs/pose_fine_patch_045_pilot` and `I:/AnatomyTracker/runs/pose_fine_patch_045_development_eval`.

Identity-equal selected-key recall (percent of valid 16×16 tissue queries):

| Proposal | Distance | Before 045 | After 500 batches | Supported key present in top 32 |
| --- | ---: | ---: | ---: | ---: |
| 019 prior top one | ≤0.5 mm | 3.2% | 4.4% | 21.8% |
| 019 prior top one | ≤1.5 mm | 27.8% | 33.0% | 79.4% |
| Physically best of eight | ≤0.5 mm | 5.8% | 8.1% | 28.1% |
| Physically best of eight | ≤1.5 mm | 44.2% | 53.0% | 99.3% |

At 1.5 mm, every appearance stratum improved on the physically best candidate: raw 43.5→48.1%, exact-black 46.9→58.9%, imperfect-brush 42.4→52.4%. Nevertheless, the predeclared criterion required at least 55% and a 15-point gain; the observed 53.0% and 8.9-point gain fail it. The physically best candidate is an oracle analysis of proposal capacity, not a model-selected plane. This is only a correspondence result; no improved physical pose or biological accuracy is claimed.

The newly measured 0.5-mm availability ceiling is more consequential than the modest training gain. The 041 key grid spans a wide off-plane search at coarse in-plane/depth spacing. Even a perfect top-32 classifier cannot select a key within 0.5 mm on most tissue queries: only 28.1% of best-candidate lists and 21.8% of top-one lists contain one. Thus continuing classifier training alone cannot reach the precision needed for electrode-site assignment. The next model stage must combine better shortlist discrimination with **continuous, local 3D refinement of the selected key**, trained from known synthetic tissue-to-CCF offsets and evaluated for actual spatial error. It should preserve a no-match option for unsupported/torn tissue. Only after stable correspondence and fit should matching quality teach the coordinate predictor, as required by the final two-task objective.

This panel has already informed 041–045 development decisions and is not independent validation. No calibrated probability, GUI replacement, animal-level expert result, or DeepSlice superiority is established.
