# Joint pose–mapping 121: frozen development result

The 8,000-update run completed normally. An independent post-run audit verified its
source, protocol, parent and checkpoint hashes, 16,000 accepted synthetic draws,
and TRAIN–DEV section/donor separation. Step 0 reproduces the parent row by row.
The fit-only gradient reaches the corrected pose, direct pose head and shared image
encoder (weighted direct-state gradient ratio 2.13% at its first eligible audit).
This establishes a learning connection, **not** that the feedback improves pose.

| Checkpoint | Synthetic selected rigid / mapped (mm) | Best of 16 rigid (mm) | Weak real coronal / sagittal (mm) |
| --- | ---: | ---: | ---: |
| Parent 111 and step 0 | 3.091 / 3.082 | 1.500 | 0.999 / 2.156 |
| Step 2,000 | 2.776 / 2.760 | 0.950 | 11.045 / 3.800 |
| Step 5,000 | 2.784 / 2.777 | 0.940 | 10.929 / 3.694 |
| Step 8,000 | 2.755 / 2.740 | 0.915 | 10.898 / 3.573 |

Synthetic figures are means over 64 independently drawn held-out sections, equal
weight for each of eight virtual plans. The real figures are donor-equal means over
64 coronal sections from six donors and 158 sagittal sections from eight donors.
Their inherited Allen affines are weak references, **not** expert truth. At step
8,000, exact-pose-given and near-pose-given mapping errors are 0.141 and 0.741 mm;
blind selection remains the dominant synthetic bottleneck (best action 0.915 mm
versus selected 2.755 mm). On the 16-section fit diagnostic, a wrong plane has the
better fit score in 8 of 15 support-matched comparisons. The fit measure is not yet
reliable evidence of pose correctness.

All trained checkpoints pass the synthetic advancement thresholds and fail both
predeclared real non-regression gates. `development_choice` is `null`: **do not
promote 121 or expose it as the new GUI model**. The coronal collapse already
appears at step 2,000, after synthetic-only updates and before real retention
begins. At that checkpoint only 8/64 coronal selected actions use the correction,
so direct pose/feature drift, rather than solely the action selector, is the
leading explanation. Sparse, 0.1-weight real pose updates every eighth step later
reduce neither TRAIN nor DEV coronal error to an acceptable level. This diagnosis
is an inference from the frozen timing and rows, not a proven causal ablation.

The next same-architecture continuation should protect real pose from its first
update, evaluate early on donor-disjoint real images, and not increase fit feedback
on the assumption that lower image–atlas fit means a better plane. Any new fit
criterion must show discrimination against anatomically plausible wrong planes.
No expert-animal accuracy, calibrated probability, electrode-region confidence,
GUI replacement, or public-benchmark superiority is established.

Frozen runs: `I:/AnatomyTracker/runs/joint_pose_map_121` and
`I:/AnatomyTracker/runs/joint_pose_map_121_dev_eval`. Evaluation summary SHA-256:
`3587858b1f73d9164f82786926fbe09c28c97859b1afa7afcd96cb1b9b52163d`.
Raw rows and all output hashes are recorded in the evaluation `completed.json`.
