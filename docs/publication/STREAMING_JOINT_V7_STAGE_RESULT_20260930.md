# Streaming joint v7 stage — frozen result (2026-09-30)

`I:/AnatomyTracker/runs/joint_v7_streaming_joint_002` exited successfully after
24,000 stage updates (30,000 total). The whole 5,197,180-parameter model and
AdamW state continued from the randomly initialized v7 lineage; no legacy
Anatomy Tracker weights, features or pseudo-labels were imported. The run used
144,000 newly generated eligible arbitrary physical planes (158,605 attempts)
from eight independent synthetic TRAIN deformation maps, plus 8,113 weakly
aligned real TRAIN sections from 58 donors. The 512 virtual synthetic identities
are global-affine variants of those eight maps, not 512 independent anatomies.

The frozen `completed.json` agrees with the 24,000 training-log lines and
158,605 synthetic-draw lines. The last log line records stage step 24,000 and
144,000 eligible optimizer planes. Schedule and identity-file SHA-256 values
match the experiment manifest; every copied source digest matches its bound
digest. Final checkpoint `joint_step_30000.pt` and the final direct readout are
present. The failed pre-update `_001` run is not part of these results.

Animal/subject-equal direct MAP readouts (mean normal error; five full-canvas
physical-point error) were:

| Stage update | Fixed synthetic TRAIN, eight maps | Synthetic DEV, four held-out maps | Real DEV, six donors, weak Allen affine |
| ---: | ---: | ---: | ---: |
| 0 | 58.10°; 12,695 µm | 56.12°; 12,399 µm | 76.61°; 10,564 µm |
| 8,000 | 47.88°; 8,318 µm | 58.01°; 12,396 µm | 3.57°; 1,004 µm |
| 16,000 | 38.53°; 6,861 µm | 54.15°; 11,954 µm | 4.04°; 700 µm |
| 24,000 | 34.99°; 5,881 µm | 54.73°; 12,065 µm | 3.61°; 657 µm |

The synthetic DEV images are old 96-pixel renders whereas training and real DEV
are 192-pixel full-canvas images. This mismatch confounds the synthetic
generalization estimate; it does not justify calling the model accurate. The
same-stage TRAIN diagnostic is still 5.9 mm, and the best of eight direct DEV
pose modes is 6.75 mm at the final readout. The real metric compares against
upstream Allen affines, not blinded expert truth, and those slices are mostly
coronal. Its apparent gain does not establish arbitrary-plane or biological
accuracy. There is no calibrated uncertainty, trained surgical-constraint
conditioning, external benchmark, or deployable model from this stage.

Next fixed diagnostic: render the already-held-out four synthetic subjects at
192 pixels from their accepted 3D deformation plans, then evaluate all 16
native fitted branches on the frozen 30,000-step checkpoint. Preserve every raw
prediction and report selected-versus-oracle errors by subject and background
mode. No public test or DeepSlice benchmark is opened for model selection.
