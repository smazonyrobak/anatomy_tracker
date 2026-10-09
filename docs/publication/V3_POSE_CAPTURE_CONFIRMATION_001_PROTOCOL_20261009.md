# v3 pose-capture confirmation 001 (frozen before panel readout)

## Question and selection boundary

The earlier 103 panel was used to diagnose raw/no-brush failures and to choose update 2,000 of `v3_one_pass_pose_capture_pilot_001`. Its observed improvement is **development evidence only**. This confirmation compares that single selected checkpoint with the exact step-zero/094 parent on new synthetic physical sections and new deformation/subject identities. Do not select a different checkpoint on this panel or use these identities for training. Do not use acquired final animals, expert pose truth, calibration data, or any public benchmark.

## Frozen panel

- Work only on `I:`. Generate eight new virtual-subject/deformation plans, IDs 10400–10407, seed 202610094101, in `I:/AnatomyTracker/data/v3_pose_capture_confirmation_plans_001`. Verify plan, original/map and animal/specimen/experiment identities are disjoint from TRAIN and all earlier DEV plans, including 103. Record source, source-data and output hashes.
- Draw 32 independent brain-intersecting physical planes per plan, seed 202610094102, uniformly across the full normal sphere with independently randomized offset, in-plane rotation and section thickness. Draw one independent v3 slide appearance/artifact realization per physical plane, seed 202610094103. Do not deliberately reuse geometry or create paired backgrounds. Reject only the existing insufficient-visible-tissue/empty-geometry cases, and retain their rejection/eligibility receipts.
- Freeze the complete 256-section panel at `I:/AnatomyTracker/data/v3_pose_capture_confirmation_panel_001` before model evaluation. Keep precise lineage, plane, appearance, artifacts, surviving-pixel CCF truth and file hashes. Ineligibles remain documented but are excluded from the metric by the same existing eligibility rule. This is a synthetic generalization test across new virtual identities, not an acquired animal-level accuracy test.

## Evaluation and gate

Evaluate only step 0 and the preselected step 2,000. Verify checkpoint and source hashes, exact step-zero equality to 094, and panel/previous-data disjointness. Give each checkpoint its own natural prior top-eight old plus top-six anchor candidate beam. Use the same frozen 094 spatial fitter, map and score; compare the identical 1,024 deterministic surviving observed pixels per eligible section. Primary metric: plan-equal mean of the within-beam **best** mapped 96-grid physical CCF distance for raw/no-brush sections, in micrometres. Also report pre-fit best, score-selected mapped error and selection regret, each appearance mode, nearest-axis family, angle stratum and per-plan counts. The best-of-14 metric diagnoses capture; score-selected error is the actual automated choice and must not be conflated with an oracle.

The selected checkpoint confirms the capture gain only if all of these are true relative to step zero: raw/no-brush best-of-14 mapped error improves by at least 0.35 mm; raw means improve in at least six of eight plans; overall score-selected mapped error does not regress; and score-selected mapped error regresses by no more than 0.20 mm in any exact-black, imperfect-brush, AP-, DV- or ML-nearest-axis group. No reused real DEV donor score enters this fresh synthetic gate. Report continuous results and failures even if the Boolean gate passes.

Passing establishes only that the pose-capture gain repeats on new synthetic identities. It does not prove adequate selected-pose accuracy, real steep-oblique transfer, calibrated probabilities, or readiness for the GUI. A failed gate stops promotion of this checkpoint; it should guide a targeted data/architecture change rather than more unqualified training or a public benchmark.
