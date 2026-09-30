# Recurrent pose-feedback pilot — frozen negative result (2026-09-30)

The corrected v8 model explicitly composes a bounded full-pose update after
each atlas-render/anatomy comparison and re-renders from that revised pose.
The first two comparisons are pose-only; later comparisons also update the
constrained curved deformation field. A learned candidate-quality head is
supervised by synthetic physical slab error. The model continued only the
whole v7 checkpoint from the current random-initialization lineage, with new
zero-initialized heads and a fresh optimizer. It imported no old Anatomy
Tracker model weights or pseudo-labels.

The six-thousand-update architecture pilot
`I:/AnatomyTracker/runs/joint_v8_feedback_pilot_001` exited cleanly. It made
24,000 fresh arbitrary-plane attempts from the eight existing TRAIN synthetic
deformation maps and used 21,763 eligible planes. The 6,000 training-log and
24,000 provenance-draw lines, last-step counters, schedule digest, all copied
source digests, and final checkpoint were checked after exit. Pose updates
became nonzero during training. These checks establish a working feedback
path, not useful generalization.

The predeclared matched 192px synthetic DEV evaluation also exited cleanly:
64 physical sections, 192 presentations from four disjoint synthetic subjects,
128 support-eligible presentations. Every 192 prediction-file digest and the
result-manifest digests match. All 16 fitted mode/reflection branches were
retained. Subject-equal, eligible readouts are below. Direct and fitted
five-point errors use the **same reflection-aware full-canvas physical metric**;
the visible-tissue centre-surface error is a separate metric.

| Input | Direct selected pose | Same branch after pose fitting | Fit-quality-selected pose | Best fitted branch (oracle) | Selected visible surface |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw background | 8,605 µm | 8,459 µm | 8,778 µm | 3,109 µm | 3,019 µm |
| Exact black outside tissue | 8,876 µm | 8,721 µm | 9,202 µm | 2,997 µm | 3,342 µm |
| Imperfect brush | 8,876 µm | 8,744 µm | 8,658 µm | 3,226 µm | 2,911 µm |

Across all 128 eligible presentations, recurrent pose fitting improved its
input branch in 100 cases, averaging 143 µm improvement. Quality ranking
changed 80 branches; it helped 36 and hurt 44, for an average 287 µm loss
relative to keeping the same fitted branch. This is **not** a successful
realization of the desired closed-loop model. The best possible branch remains
about 3 mm from synthetic reference anatomy, and the selected branch about
9 mm. Neither uncertainty nor surgical conditioning is calibrated/trained,
and none of this is an animal-level or DeepSlice benchmark.

Decision: preserve v7/v8 pilots as diagnostic baselines. The next training
stage requires substantially more independent local 3D anatomy maps and
fresh physical planes, plus candidate-quality supervision spanning all pose
modes/reflections rather than one geometry-best plus one distractor. Evaluate
equal-subject improvement and safe candidate selection on held-out synthetic
anatomy before broader real-data training or any public comparison. The
TRAIN-only real image expansion remains separate, with upstream Allen affines
explicitly treated as weak labels.
