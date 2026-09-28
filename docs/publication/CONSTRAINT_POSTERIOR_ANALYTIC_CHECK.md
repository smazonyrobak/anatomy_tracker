# Constraint-conditioned posterior primitive: analytic check

2026-09-28. `training/arbitrary_plane_constraint_posterior.py` is **not wired into
the model, trained, calibrated, or enabled in the GUI**. It computes conditional
physical evidence for an explicitly supplied trajectory hypothesis; it does not
estimate an unknown ray or claim calibrated region probabilities.

Input image log probabilities retain the full cell/representation table.
Frames and marks have an explicit section axis. Marks map through the cell warp,
then the exact representation affine, then the physical frame. Mark likelihoods
sum over sections; entry/angle surgical factors are applied once per animal
hypothesis. For multiple sections, hypothesis columns must already represent
coherent joint section assignments, not unrelated per-section marginals.
Recurrent updates must recompute against the original image prior rather than
repeatedly feeding back the conditioned posterior and counting evidence again.

A CPU float64 analytic experiment used two physical plane hypotheses, identity
and mirrored representations, two sections, an explicit shared finite ray and a
nonzero source-to-fixed warp. Results:

- The intended physical mode received posterior probability 1.0 at displayed
  precision; this is a constructed geometry check, not an accuracy estimate.
- With all physical evidence absent, the original posterior tensor is returned
  unchanged, including object identity.
- Mirrored mark positions remained separate at ML 3750 and 4225 um, confirming
  warp-before-reflection order rather than coordinate averaging.
- Finite nonzero gradient L1 norms reached pose (8.9959) and warp (0.4082), and
  also the explicit ray entry/direction.
- Shared surgical log likelihood was exactly equal with one and two sections;
  only independent mark evidence was accumulated across sections.
- Infeasible hard constraints preserved an all-infeasible flag and all `-inf`
  posterior values without NaNs. Section-specific physical plane bounds worked.

The flat experiment is saved at
`I:/AnatomyTracker/tmp/constraint_posterior_manual_20260928.py`. No active run
outputs were accessed. Next integration must preserve ray uncertainty through
joint inference/marginalization, avoid duplicate marks/factors, and evaluate
conditional calibration on unseen animals before making probability claims.
