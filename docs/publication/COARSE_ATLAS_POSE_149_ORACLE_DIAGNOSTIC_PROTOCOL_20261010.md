# 149: why 148 did not correct a nearby plane

The frozen 148 synthetic DEV gate failed. This is a diagnostic on those already-open development sections, not a new model selection set or evidence of real-section accuracy. No learning, parameter tuning, or public/final benchmark is permitted.

For every eligible v4/v3 frozen DEV section, reproduce the frozen 132 truth-best beam branch and its reflection. Apply the exact 148 weighted affine chart-to-CCF solve (ridge 24/12/12) at the 24×24 source grid, then score corrected rigid plane error at *all* valid native-256 tissue sites. Use the frozen full-intensity 148 checkpoint at step 6000 to supply its prediction, confidence, and supported key lattice. Compare these four fits, changing only one component at a time:

1. Learned expected CCF correspondence with learned visibility×match-mass confidence: reproduce the deployed 148 fit and check it against the frozen 148 evaluation.
2. The true observed-pixel CCF map after synthetic deformation and artifacts, snapped to the nearest *supported, 3-mm-reachable* 148 lattice key, with the learned confidence unchanged. This isolates matching error while retaining learned weighting.
3. The same nearest valid oracle key, but with ideal observed-validity weights. This tests the additional confidence/visibility error. Report unreachable sites and keep every section in the denominator; an unreachable site receives zero fit weight, not a filtered evaluation score.
4. The true continuous observed-pixel CCF map with ideal observed-validity weights. This tests whether a global least-squares plane fit is confounded by local deformation even with perfect continuous correspondence.

An optional exact rigid-CCF correspondence fit with ideal weights isolates the ridge solver limit. A lattice without atlas support is only an optimistic discretization bound, not the main comparison.

Also evaluate the exact target state for the same conditions. In all cases the target to score is the original rigid plane, not the deformed pixel map; do not use an artifact-selected affine refit. Report section-equal and synthetic-plan-equal means, key reachability, and the gap between these conditions and trained 148 at checkpoints 3000/6000. This is a mechanistic upper-bound analysis, not an implementable inference path. Retain all failed/invalid cases in denominators. Do not use this panel to claim a calibrated uncertainty or choose hyperparameters by repeated tuning.
