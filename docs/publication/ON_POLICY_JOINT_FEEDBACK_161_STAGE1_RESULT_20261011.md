# Joint fitting-feedback 161 stage 1: bounded TRAIN pilot failed its gate

The [pre-run protocol](ON_POLICY_JOINT_FEEDBACK_161_STAGE1_PILOT_20261011.md) froze a conditional test of whether the existing atlas-conditioned fitter can learn a better near-versus-wrong difficulty score before allowing it to update the direct pose model. This experiment did **not** train pose feedback. Both arms started from the same scratch-trained parent-132 checkpoint and used the same 192 distinct near-eligible synthetic TRAIN sections in the same order, 192 one-section updates each. The full arm retained atlas intensity; the control zeroed atlas intensity but preserved atlas support. Both inherited parent weights that had already seen full atlas intensity, so the comparison tests intensity access during this continuation, not independent from-scratch architecture effects.

The frozen inner panel contained one support-matched near/wrong pair from each of 64 distinct TRAIN physical sections on disjoint deformation bases. The pairs were selected from blind model branches and differed in known rigid error by at least 1 mm while having atlas-support Dice at least 0.85, support-area gap at most 0.08, and at least 32 common cells. One decision per section was counted. Checkpoints at 0, 96, and 192 updates were selected separately per arm under the predeclared map-retention limits.

| Inner result | Full atlas intensity | Atlas support only |
| --- | ---: | ---: |
| Step 0 near wins | 47/64 (73.44%) | 41/64 (64.06%) |
| Selected checkpoint | 96 updates | 192 updates |
| Selected near wins | 50/64 (78.13%) | 46/64 (71.88%) |
| Selected near/exact map error | 0.802/0.155 mm | 0.817/0.168 mm |
| Selected exact-plane rigid drift | 0.0027 mm | 0.0030 mm |

The selected full arm gained only 3/64 = 4.69 percentage points over its own step 0, below the required five points; it beat the selected support-only arm by 4/64 = 6.25 points, below the required ten. Map retention passed in both arms. The stage-1 gate therefore **failed**. The 192-update full arm fell back to 48/64 near wins; the result does not support extending the same optimizer just because more updates are possible.

Two fixed-pair out-of-distribution probes show that the selected full model does use image evidence: shifting atlas intensity by 100 atlas voxels along AP while keeping support fixed reduced near wins from 50 to 35/64, and zeroing the source visual feature tensor reduced them to 40/64. These perturbations are not a clean anatomical-information control or a rescue of the failed full-versus-support gate. They also do not demonstrate that the score improves final plane localization.

The attempt log is important: 232/420 TRAIN attempts had less than 8% valid tissue. Of those, 58 were eligible and 27 had a near branch, but **none** supplied a support-matched training pair. The 161 pilot therefore says nothing about repairing the sparse-cut candidate-coverage limitation independently measured in [diagnostic 163](SPARSE_PLANE_CAPTURE_163_TRAIN_RESULT_20261011.md). At the selected checkpoint, the near-branch map error was still approximately 0.8 mm even though an exact supplied plane mapped at approximately 0.16 mm; this is a clear localization/fitting gap, not a deployable alignment.

All declared output SHA-256 values and the disjoint physical-section manifests were independently checked after exit. Frozen run: `I:/AnatomyTracker/runs/joint_feedback_161_stage1_train_pilot_001`; completed receipt SHA-256 `ceda6984c3e50f5721cdd7e599f913946a37dbabe8ffcb831a22837ff7dd11c`; summary SHA-256 `6bc00ee569bbc9609605d329f36eef199c091403c59fd49252377171de4fe8e8`. The source is `training/train_joint_feedback_161_stage1_pilot.py`. No DEV or final animal, real anatomical ground truth, public benchmark, calibrated probability, or GUI qualification was used.

**Decision:** retain both checkpoints and this result for analysis only. Do not run the predeclared stage-2 live-feedback-gradient comparison, promote the fitter, or claim arbitrary-plane success. The next experiment should target the actual bottlenecks: anatomy-specific matching under support-similar alternatives and broad enough pose proposals when little tissue is visible. Any revised experiment needs a fresh frozen TRAIN-inner panel and an explicit support-only comparison rather than reusing these 64 cases to tune until a threshold is crossed.
