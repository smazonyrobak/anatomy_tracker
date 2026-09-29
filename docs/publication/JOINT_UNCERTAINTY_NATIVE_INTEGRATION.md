# Opt-in native joint uncertainty — 2026-09-29

**Scope after the curved-ribbon extension:** this historical35-coordinate head
models the older planar/SVF branch. It has not been adapted to the new3D centre
surface and through-thickness director, and is not enabled in the native ribbon
learning control. Do not export it as ribbon or electrode-site confidence.
Correlated ribbon/ray uncertainty and biological-animal calibration remain
required before any90% region-probability claim.

The joint v6 model now has an optional, randomly initialized covariance head on
the shared recurrent updater's **final context for each selected cell and each
raster representation**. Only spatial dimensions are pooled. Reflection modes
are not averaged into one uncertainty prediction. Set `joint_uncertainty_rank=4`
to enable it; `None` is the default and creates no additional parameters or
buffers, consumes no additional random numbers, and leaves forward output
unchanged. Existing checkpoints still load strictly with the default disabled.

This is a native forward/learning-objective integration, **not a trained or
calibrated uncertainty model**. Current experiment runners and GUI are unchanged.

## Distribution and supervision

`refined_output['joint_uncertainty']` retains `(B,K,R)` component identity, exact
representation affines, selected-cell probabilities, and unresolved omitted
catalogue mass. Its 35 coordinates are local rotation (3 radians), local
translation (3 micrometres), log-basis diagonals (2), shear (1), and smooth SVF
coefficients (26 pixels). The FP32 head predicts diagonal SD `s` and low-rank
loadings `U`: normalized covariance is `diag(s**2) + U @ U.T`.

Physical coordinate scales are a saved checkpoint buffer, defaulting to
`(0.1,0.1,0.1,500,500,500,0.1,0.1,0.1) + (1,)*26`. With diagonal matrix `D`
containing those scales, native covariance is `D @ covariance @ D`. No implicit
unit conversion is applied. The default SD is approximately one normalized unit;
small nonzero loading weights allow covariance gradients at initialization.

`joint_uncertainty_nll(distribution, target_frame, target_velocity, basis, observed)`
computes exact selected-coordinate Gaussian marginals. `observed` has shape
`(B,K,R,35)` and must identify only truth-compatible cells/representations and
identified coordinates. Use the returned `observed_coordinate_count > 0` when
averaging component NLLs; missing components do not become zero-error targets.
The native-coordinate NLL adds the observed-dimension log-scale Jacobian to the
normalized-coordinate NLL. Frame targets must be in the same local canonical
chart, and velocity targets in the same source-raster gauge.

The mean frame and mean velocity are detached in the residual calculation by
default. Covariance gradients still reach the shared recurrent context. This
removes direct NLL pressure on the deterministic residual, but **does not prove
point-estimate non-inferiority**: shared-feature changes can alter point estimates,
so a later paired accuracy ablation remains required.

The current decoder explicitly projects with uniform canvas weights, despite
also predicting a support mask. Its gauge therefore matches the existing 26D
basis. Only identifiable, fully observed synthetic velocity fields may supervise
those coefficients. Dense-censored and pose-only rows are automatically excluded
from SVF supervision. Returned `omitted_velocity_energy_px2` quantifies residual
variation outside the smooth basis; this is not a distribution over every
possible nonrigid field or an estimator from partial tissue support.

## Sampling and limitations

`sample_joint_uncertainty(distribution, basis, sample_count)` draws the physical
cell (including the unresolved tail), its conditional raster representation, and
then the correlated 35D continuous residual. A section's selected representation
is retained for all its marks: warp first, representation affine second, then
the physical frame. Unresolved-tail draws have indices `-1` and NaN geometry;
their probability is not redistributed among rendered cells. Teacher-forced
training selections cannot be sampled as inference posteriors.

The deterministic frame and warp are still shared across a cell's representations;
**only their covariance is representation-specific**. The earlier deterministic
pose/deformation averaging limitation has not been solved by this head. Sampling
returns perturbed SVFs before integration, flags departures from the local
rotation chart, and performs no silent clipping, topology rejection or redraw.
Such samples still need explicit physical/topology assessment before electrode
probabilities. Surgical factors/ray marginalization, animal-level calibration,
and GUI site-cloud presentation are not wired into this integration.

## Focused CPU evidence

One 16x16 analytic recurrent example, two cells, two exact reflection affines,
rank four, and 4,000 discrete/continuous draws exercised the existing renderer,
recurrent updater, new head, objective and sampler. The completed historical
baseline checkpoint was read only to verify strict disabled-mode compatibility;
no active run or GPU was accessed. No optimizer step or training was performed.

- Disabled state-dict keys, initial tensor values and RNG state exactly matched
  the pre-integration model at `37c15a2`; the historical checkpoint loaded strictly.
- SD shape `(1,2,2,35)` and factor shape `(1,2,2,35,4)` retained reflections.
  Distinct reflected contexts differed by up to `0.081511`; predicted SDs differed
  by `7.1406e-5` at random initialization.
- All backward gradients were finite. Head weight gradient L1 was `9.79366`;
  final shared-context gradient L1 was `0.0712266`; recurrent candidate weight
  gradient L1 was `0.815131`. Direct mean frame/velocity gradients were absent.
- Censored all-missing NLL was exactly zero; in-basis omitted velocity energy
  was `1.8960e-14 px²`.
- Maximum empirical cell/representation frequency error was `0.01225`.
  Unresolved frequency was `0.25325` for assigned mass `0.25`; all tail geometry
  was NaN and every resolved affine was an exact selected reflection, not an average.

Script: `I:\AnatomyTracker\tmp\joint_uncertainty_native_20260928.py`.
Result: `I:\AnatomyTracker\tmp\joint_uncertainty_native_20260928.json` (rerun September 29).
SHA-256 receipts:

- Script: `1708549ad3aeb72708fde8fa46e038426c1e45588a98aa881ae97c04cb1db2db`
- Result: `3095b0f0c9fe6a7b90648904c50915dbe5396cf1833d5797c3e1cbcc8df806f2`
- Joint model: `2a6afc41b8cf13fa1f297b9a1fa30eb8703e81dae28917624bca5d5665d70aae`
- Uncertainty module: `733b5c1129a14ce1f8ddb300cacd93b781f5e1f6594169335071c61445678e09`
