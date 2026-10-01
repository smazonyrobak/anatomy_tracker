# One-shot joint curriculum: frozen internal readout (1 October 2026)

Run `I:/AnatomyTracker/runs/one_shot_joint_pose_curriculum_002` used code commit `85c4b04`, ten thousand four-section batches (40,000 fresh synthetic plane presentations), and the random-start one-shot lineage's initial batch-500 checkpoint. The atlas-fitting gradient was **not** active: this stage trained direct pose and correct-pose local mapping. The process exited and `completed.json` reports all 10,000 batches. Raw training, draw, development and checkpoint files are retained on `I:`.

The run evaluated 64 synthetic development sections from four held-out synthetic deformations and 64 real sections from six held-out Allen donors with **weak**, pre-existing alignment labels. Synthetic deformation identities are not biological animals. The original in-run readout used 192-pixel development images although training used 256 pixels. The independent frozen evaluation `I:/AnatomyTracker/runs/one_shot_joint_pose_curriculum_002_resized_eval` resized development images to 256 and did not rescue performance:

| 256-pixel readout, donor-equal mean | Start | Batch 8,000 | Batch 10,000 |
| --- | ---: | ---: | ---: |
| Synthetic selected five-point error | 13.83 mm | 12.70 mm | 12.53 mm |
| Synthetic oracle of 16 candidates | 7.78 mm | 7.39 mm | 7.28 mm |
| Synthetic selected plane-normal error | 61.0° | 53.0° | 47.7° |
| Real weak-Allen selected five-point error | 15.07 mm | 11.64 mm | 11.87 mm |
| Real weak-Allen oracle of 16 candidates | 6.77 mm | 5.58 mm | 5.28 mm |
| Real weak-Allen selected plane-normal error | 79.0° | 29.4° | 20.5° |

Training-set best-mode five-point error averaged 6.00 mm in the first thousand batches and 4.83 mm in the last, while the held-out selected result stayed poor. The best-development checkpoint is batch 8,000. Correct-pose mapping on visible synthetic development tissue was 59.28 µm with zero added warp and 58.76 µm at batch 10,000: the fitter has not learned a useful held-out improvement. This is not a deployable model or evidence of calibrated uncertainty.

Two focused diagnostics change the next action. First, `I:/AnatomyTracker/runs/one_shot_fit_rank_001` shows the true plane had a lower atlas mismatch than AP shifts of ±250, ±500 and ±1,000 µm on all 43 eligible synthetic development sections (mean mismatch 0.330 at truth versus 0.605–0.843 at the shifted planes). But `I:/AnatomyTracker/runs/one_shot_candidate_fit_001` shows that this score **does not** yet select well among the model's often distant 16 candidates: 11.68 mm fit-selected versus 11.58 mm prior-selected and 5.78 mm oracle on the same 43 sections. Thus atlas fitting has a useful local signal, not a reliable global selector at the present pose accuracy. Fitting-to-pose training remains required, but should not be switched on blindly at this stage.

Second, the synthetic augmentation sometimes shifts tissue by a global affine component while supplying the pre-augmentation plane as its target, even though the model explicitly removes global affine terms from its local warp. On 222 eligible fresh draws, the mean visible-pixel excluded component was 28.27 µm, reaching 56.28 µm in the highest warp-strength quartile (`one_shot_warp_gauge_001`). Commit `cfc3a2b` refits the global pose to the observed dense target and preserves the source pose in provenance; the same frozen-seed diagnostic then measured 0.001 µm excluded component (`one_shot_warp_gauge_002`). This repairs the training target's pose/warp separation, not the current model's 12 mm alignment error.

Next stage: continue the same model from the frozen batch-8,000 checkpoint with the corrected synthetic generator and a donor-balanced stream of distinct real TRAIN sections. The prepared 20,000-batch stage presents three freshly sampled synthetic planes plus one unique real training section per batch, with no public benchmark or calibration data. Reassess frozen checkpoints on separate development donors and correctly scaled inputs before deciding when predicted planes are close enough for joint fitting feedback.
