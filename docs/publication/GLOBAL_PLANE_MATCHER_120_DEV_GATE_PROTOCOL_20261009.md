# 120 global plane matcher: predeclared synthetic DEV gate

Status: specified before execution; measurement-site amendment after the first evaluator stopped before generating output. The original fully-valid 16×16-cell rule left zero sites on at least one artifact-heavy section. No checkpoint outcome was read or selected. This is a head-only feasibility gate, not deployment, calibration, untouched confirmation, real-animal validation, or a public benchmark.

## Frozen inputs and selection

- Require the completed 512-update paired 120 TRAIN run and verify its config, draws, training log, source and checkpoints (0, 256, 512) by SHA256 before loading any checkpoint. Load the exact frozen 111 step-1959 parent.
- Use the existing v3 pose-capture confirmation panel only as reused synthetic DEV. Exclude all 64 section IDs in the completed 118 assay, irrespective of its scoring status. Within each of the eight synthetic plans, take the next eight eligible records in ascending SHA256(section_id) order. Record the ordered IDs, panel-file hashes, exclusion hash, panel/plan receipts, and identity-disjointness from used 120 TRAIN draws.
- Reproduce the 120 blind beam without truth: the frozen 111 prior's top eight base branches and top six anchor branches, then two unused anchors selected for maximal predicted-normal diversity. Thus the original 14 branches remain intact. Neither the true nor a near-true state is a blind candidate.

## Measurements

- At 256 valid observed pixels per section, sampled with replacement only if necessary by a SHA256(section_id)-seeded generator, primary rigid pose error is mean CCF Euclidean distance to `target_state`; separately report rigid-only observed-pixel error to `target_centre_um` (not the predicted deformable map). The same frozen pixels are used for every checkpoint, arm and action. Also report mean rigid CCF error at the fixed four corners and centre. All reported errors are millimetres.
- `target_state` is the observed-image affine gauge fitted to the warped synthetic coordinate field; it is not the pristine pre-warp cutting plane. The two targets answer different questions and must not be conflated.
- At each checkpoint and arm, report the original 111 prior-selected action on the first 14 slots, the same-beam input-16 selected action, corrected-16 selected action, and joint two-action selected action. Also report physically best corrected-16 as a truth-only oracle diagnostic, never as a selectable action. Compare full atlas intensity with zero-intensity/support-only at the same checkpoint, section, beam, and parent features.
- Give per-plan and plan-equal means; stratify raw, imperfect-brush, exact-black, angle (<15, 15–30, 30–45, ≥45 degrees; also ≥30), and truth-plane atlas support fraction (<.25, .25–.50, ≥.50). Support strata use truth only for diagnostic grouping, never selection. Define a near case as original-14 best five-point rigid error ≤1.5 mm, and report joint-action nonregression relative to the 111 prior at +.20 mm tolerance for rigid-grid and five-point error.

## Gate fixed in advance

Select one DEV checkpoint by lowest atlas-arm plan-equal joint-action rigid-grid error among 0, 256, 512, tie to earlier step. At that checkpoint the exploratory gate passes only if all hold:

1. Atlas joint-action rigid-grid error improves on the 111 prior by ≥.20 mm plan-equal.
2. Atlas joint-action rigid-grid error improves on support-only by ≥.20 mm plan-equal, paired on the same cases.
3. Atlas truth-only best corrected-16 rigid-grid error improves on original best-14 input by ≥.50 mm plan-equal, and in at least six of eight plans.
4. Near-case joint rigid-grid and five-point plan-equal error each remain within +.20 mm of the 111 prior. If no near cases occur, this condition fails as unevaluable.
5. Raw, imperfect-brush, ≥30-degree, and each atlas-support bin with at least eight cases retain joint rigid-grid error within +.20 mm of the 111 prior plan-equal.

Report every component, including failures and sparse strata. A pass licenses only a fresh, separately declared confirmation experiment and later joint/fitting and real-animal checks.
