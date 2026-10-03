# Frozen normal-conditioned cost-volume updater: 077 result

The predeclared 077 pilot trained a two-pass, shared-weight spatial updater on the frozen 059 model's **14 actual blind pose proposals**. It rendered a finite-thickness atlas context around each proposal and correlated that context with the image before predicting local pose corrections. Synthetic truth supplied a training target but never created or ranked a proposal at inference. The protocol and implementation are `NORMAL_CONDITIONED_COST_VOLUME_077_PROTOCOL_20261004.md` and `training/normal_conditioned_cost_volume_077.py`.

Training completed 2,000 batches of two independently sampled synthetic sections and one weak-real TRAIN section. The frozen record contains 4,000 accepted unique synthetic physical sections (406 rejected draws) and 2,000 distinct weak-real TRAIN draws, with source, schedule, parent, checkpoint, and output hashes bound in the receipts. All development files were on `I:`. There was no TRAIN/DEV identity overlap and no public or final-test data was used. Checkpoint zero reproduced the frozen 059/075 parent metrics to numerical precision.

On the fixed 246-section/eight-synthetic-identity DEV panel, the identity-equal visible-tissue error was:

| Checkpoint | Selected pose, mm | Truth-best of 14, mm |
| --- | ---: | ---: |
| Frozen parent / step 0 | 2.656 | 1.022 |
| Step 1,000, two corrections | 2.645 | 1.135 |
| Step 2,000, first correction | 2.632 | 1.064 |
| Step 2,000, two corrections | **2.640** | **1.149** |

The best-of-14 column is a truth oracle, **not** an available model output. The final selected gain was only 0.016 mm while the oracle worsened by 0.127 mm. On the 172 sections with an originally near-true candidate (parent best-of-14 ≤1 mm), best-of-14 worsened from 0.690 to 0.839 mm. The second correction was weaker than the first, consistent with drift rather than useful feedback. Exact-black, imperfect-brush, raw-background, and low-support selected errors at step 2,000 were 2.537, 2.531, 2.762, and 3.530 mm; none showed a material improvement. On 64 weak-Allen real DEV sections from six donors, selected five-point error changed from 0.694 to 0.680 mm, but those inherited affines are not expert anatomical truth.

The predeclared advancement gate **failed**: the final selected error exceeded 2.40 mm, the final truth-best-of-14 exceeded 0.82 mm, and the near-true subset worsened by more than the allowed 0.10 mm. Donor and appearance/support nonregression conditions passed. Reject this updater for scaling, release, or GUI integration. The result does not prove that rendering-based feedback cannot work; it shows this particular coarse spatial signal and supervision did not capture or preserve accurate pose on the present panel.

The training audit found one small weak-real target-convention mismatch: reflected 192-pixel source coordinates used 191/192, whereas the resized 077 chart used 255/256. That shifts the weak-real training target by about 16 µm over a 12-mm field. Synthetic targets and evaluation were unaffected; the discrepancy is much smaller than the observed millimetre-scale failure, but the frozen checkpoint was not altered after discovery.

The independent evaluation completed with exactly 930 rows (246 synthetic plus 64 weak real at each of three checkpoints). Raw rows, summary, input-panel, preregistered protocol, source, frozen parent, and all checkpoint hashes matched their receipts; row-level metrics were finite and the identity-equal means were recomputed from the rows. Output rows SHA-256: `684dea7545fb6ace9ea6acaaa994cd2e54b4f090c00db61e640f301fc87ab9f7`; summary SHA-256: `d10281bf6521ff76ba6d98fb3bf5b9b06072b7d33bebcf45aef8b9020a2da212`. Frozen training and evaluation outputs are `I:/AnatomyTracker/runs/normal_conditioned_cost_volume_077_pilot` and `I:/AnatomyTracker/runs/normal_conditioned_cost_volume_077_development_eval`.

The next decision should address the persistent **global pose-selection/capture and data-identifiability problem**, not simply add another local correction stage. This was a development-only synthetic/weak-label test: no calibrated uncertainty, biological animal-level final test, GUI promotion, or public DeepSlice comparison was performed.
