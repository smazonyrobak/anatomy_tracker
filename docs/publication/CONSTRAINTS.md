# Constraint semantics and validation

Constraints are deterministic inference over the model's scored pose candidates. They are not hidden training labels and do not change model weights. Unconstrained inference remains the primary automatic result.

## AP interval

An enabled interval is a hard mask over bregma-centred AP coordinates. Candidates outside it are never evaluated as feasible and a final answer is never clipped onto the boundary. The interface normalizes entry order (`from`/`to`) but preserves the numerical closed interval. A boundary optimum is reported as potentially search-limited.

## Partial anterior--posterior order

Only checked slices participate. Their user-defined order imposes strict anterior-to-posterior inequalities. Unchecked slices have no order relation. Duplicate filenames or displayed numbering do not define order. The joint solver optimizes the sum of slice pose energies subject to only the selected relations.

## Common tilt

`Auto-align all` optimizes one exact L--R/D--V pair shared by all eligible outlined slices, while each slice retains its own AP and in-plane similarity/deformation. The displayed 3-D planes must therefore be parallel up to numerical tolerance. Independent alignment retains per-slice tilts.

## Surgical constraints

Surgical inputs include a bregma-centred entry location and radius, attack angle and tolerance, physical depth limits, probe geometry and raw histology observations. They constrain feasible slice pose because the transformed observations must admit a physical trajectory through the allowed cortical entry disk and angular interval. After pose/warp mapping, the fitted trajectory remains constrained to the same feasible set; it is not an unconstrained regression followed by a warning.

The solver must not multiply runtime through an unconstrained exhaustive search. Feasible pose/tilt ranges are precomputed or pruned, then combined with image-derived energies. A hard surgical constraint cannot be exceeded by rounding tolerance.

## Infeasibility

If no solution satisfies all enabled hard constraints, the solver returns:

- the conflicting constraint set;
- minimum violation for each constraint family;
- the best unconstrained/soft diagnostic candidate only when explicitly labelled and not installed as the constrained result.

Constraints are never silently relaxed. A separate user action may convert selected hard constraints into documented probabilistic priors.

## Constraint benchmark

Synthetic cases include feasible and intentionally infeasible combinations. Report:

- unconstrained accuracy;
- incremental accuracy and runtime for every constraint family;
- combinations of AP/order/common-tilt/surgical constraints;
- hard-satisfaction rate, required to be 100%;
- false infeasibility and false feasibility rates;
- result invariance when constraints are disabled;
- effect of incorrect but feasible user information.

The number and type of user-provided constraints are disclosed for every assisted comparison.
