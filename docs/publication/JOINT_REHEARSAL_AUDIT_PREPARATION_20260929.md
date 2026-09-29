# Rehearsal003 endpoint audit: prepared, not run

`training/audit_joint_v6_local_refinement.py` now targets
`I:/AnatomyTracker/runs/joint_v6_joint_rehearsal_003`, compares local results with
frozen coordinate002 and its hashed audit, and writes only to the separate
`joint_v6_joint_rehearsal_audit_003` directory. The coordinate002 auditor remains
preserved in commit `8b4d58f`. This preparation does not access the active run;
execute only after its owning terminal confirms exit.

The existing independent NumPy local geometry, reflection-after-pullback,
pre-reflection topology, row/perturbation/eligibility, full-parent and frozen
parameter checks are retained. Compare exact local schedules, model kwargs,
initial tensors and archived `local_pass` source with coordinate002. Extra
source/shared/proposal trainable prefixes are the declared intervention, not
an expected equality failure. Check replay losses, one applied update per row
schedule step, copied original replay receipts/IDs and checkpoint metadata.

New proposal readout uses only saved step0/4000 raw probabilities and original
prepared truth. Sixteen-row float64 blocks independently recompute normalization,
exact ranks, full-cell and normal-marginal NLL, MAP normal/frame/physical offset
errors and angular capture. Frame axes are reconstructed in NumPy from stored
12D states. No GPU, model inference or full-gallery feature recomputation occurs.

The [predeclared retention gate](JOINT_GLOBAL_REHEARSAL_PROTOCOL_20260929.md)
compares this run's FP32 endpoints, not historical AMP predictions. Positive
proposal-weight intersections are rebuilt separately for every mode. Eligible
rows are averaged within each organizational group, then groups equally.
All-row and censored strata remain reported but cannot determine retention.
Overall tolerances are +0.10 nat NLL, +1 degree MAP normal, -0.02 exact top128,
and at most 1.05 times initial frame/offset errors; per-mode bounds are +0.20,
+2, -0.04 and 1.10, respectively. Every populated bound must pass.

Local criteria remain 20% landmark/CCF reduction, 10% pullback reduction,
non-worsening normal and no valid-tissue folds, relative to paired geometric
starts. Both gates and any local mode regressions are reported independently.
An integrity-valid negative experiment may exit0: failed scientific gates do
not throw away its evidence. Neither a pass nor this prepared source establishes
global adequacy, calibrated uncertainty, biological generalization or shipping.

Preparation verification: source AST parsing and whitespace review only;
the auditor was not imported or executed and no rehearsal003 output was opened.
