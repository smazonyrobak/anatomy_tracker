# 222 pose-selection ablation: do not promote

The [frozen protocol](POSE_SELECTION_222_PROTOCOL_20261011.md) compared a direct near-best candidate-ranking loss with an otherwise unchanged control. Both arms continued the same randomly initialized project-lineage 221 framed checkpoint for 400 updates. The independent training audit found 400 finite, consecutive updates per arm, 1,200 distinct synthetic physical-section IDs per arm (800 new framed, 400 legacy), no IDs shared between arms, and exact source, protocol, parent, draw, checkpoint, and acquired-TRAIN provenance bindings. AL1A was excluded from gradients. Both runs and every evaluation artifact are on I:.

| Fixed TRAIN-inner readout, median error in mm | 221 parent | 222 control 400 | 222 ranking 400 |
| --- | ---: | ---: | ---: |
| Original 64 framed, top choice, dense tissue | 3.450 | 3.491 | 3.255 |
| Original 64 framed, top choice, exact affine | 3.424 | 3.461 | 3.249 |
| Original 64 framed, best of 160, dense tissue | 1.474 | 1.424 | 1.615 |
| Fresh independent 64 framed, top choice, dense tissue | 3.522 | 3.073 | 3.462 |
| Fresh independent 64 framed, best of 160, dense tissue | 1.702 | 1.620 | 1.609 |
| Original 64 unframed, top choice, dense tissue | 2.794 | 2.680 | 2.864 |
| Legacy 32, top choice, dense tissue | 1.385 | 1.312 | 1.353 |

The preregistered primary gate **failed**. The ranking arm's original-framed top-choice improvement versus control was 0.236 mm, below the required 0.300 mm. Its original-framed best-of-160 exact-affine and dense errors worsened by 0.160 and 0.191 mm, both beyond the allowed 0.100 mm. Top-16 dense capture below 2 mm fell from 45/64 to 44/64. The extra fresh framed panel, sampled independently of both training streams and the original fixed readout, showed a **0.389 mm regression** in top-choice dense error versus control. Both arm step-zero checkpoints reproduced the parent readout exactly. All 64 first choices in each framed panel remained from the 16 free modes; the 64 orientation-anchor modes held about one-quarter of total probability mass but did not win a first choice. The ranking loss is therefore not evidence that the anchor candidate-selection problem is solved.

On 52 held-out AL1A TRAIN sections, median weak human-annotation five-point error was 1.338 mm at the parent, 1.304 mm for control, and 1.319 mm for ranking. These are weak references, not exact physical plane truth. On older distinct TRAIN donors, donor-equal weak five-point error was 0.565/1.352 mm for parent coronal/sagittal, 0.627/1.454 mm for control, and 0.528/1.400 mm for ranking. The ranking arm did not cause a large old-TRAIN collapse, but neither human guard rescues the failed synthetic gate or establishes arbitrary-oblique biological accuracy.

Both arms used independently drawn training streams rather than exactly matched image draws, so their difference cannot be attributed solely to the new loss. The conflicting original/fresh framed results are a further reason not to scale, select, calibrate, benchmark publicly, or integrate this checkpoint. The 221 framed-400 checkpoint remains the comparison parent, **not** a qualified deployable model. No DEV/CAL/FINAL animal split or DeepSlice benchmark was used.

The next targeted work should address the visible synthetic/acquired texture gap, low-support slices, and the persistent candidate-family ranking failure with a small, frozen, genuinely fresh TRAIN-inner comparison before greater training exposure. Do not keep tuning the 222 hinge on the original 64 cases. A probabilistic atlas-plane prediction and local tissue mapping still need to work jointly in the eventual model; this failed ablation is not a rejection of that goal.

Frozen artifacts: `I:/AnatomyTracker/runs/pose_selection_222_{control,intervention}_train_001`, `I:/AnatomyTracker/runs/pose_selection_222_paired_readout_001` (224 cases, 1,568 rows), `I:/AnatomyTracker/runs/pose_selection_222_AL1A_train_guard_001` (52 sections, 364 rows), and `I:/AnatomyTracker/runs/pose_selection_222_old_train_guard_001` (88 sections, 616 rows). All three evaluation output inventories and their evaluator-source hashes were independently checked against completion/config receipts after exit, with zero mismatches. The main readout rows SHA-256 is `fb2fd4a5867ab7d90868d31cfc19c486f38aee6ea3ae3cad40c50fd433dc94e1`.
