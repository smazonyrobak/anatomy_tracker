# Constraint semantics and validation

Constraints are deterministic inference over the model's scored pose candidates. They are not hidden training labels and do not change model weights. Unconstrained inference remains the primary automatic result.

## Plane domain and legacy coronal AP interval

The deployed AP interval is a legacy coronal-chart constraint. An enabled
interval is a hard mask over bregma-centred AP coordinates; candidates outside
it are never evaluated as feasible and a final answer is never clipped onto the
boundary. The interface normalizes entry order (`from`/`to`) but preserves the
numerical closed interval. A boundary optimum is reported as potentially
search-limited. The future arbitrary-plane solver instead masks a declared
domain in full normal/physical-offset/frame or O/U/V geometry; an AP scalar
cannot define sagittal or horizontal plane position.

## Partial stack order

The deployed coronal rule lets checked slices impose strict
anterior-to-posterior AP inequalities. For an arbitrary-plane stack, the user
or acquisition metadata declares a stack normal and checked slices instead
impose strict order on signed physical offsets along that normal. Unchecked
slices have no order relation. Duplicate filenames or displayed numbering do
not define order. The joint solver optimizes the sum of slice pose energies
subject only to the selected relations.

## Common tilt

The existing L--R/D--V-pair rule is a legacy coronal-chart constraint only; it
cannot represent sagittal, horizontal or general oblique stacks. Future
arbitrary-plane `Auto-align all` optimizes one shared antipodal 3-D plane normal
in the full frame/O/U/V geometry, while each slice retains its own physical
normal offset, in-plane frame/basis and deformation. The displayed 3-D planes
must therefore be parallel up to numerical tolerance without forcing a common
roll or scale. Independent alignment retains a separate full plane normal per
slice.

## Surgical constraints

Surgical inputs include a bregma-centred entry location and radius, attack angle and tolerance, physical depth limits, probe geometry and raw histology observations. They constrain feasible slice pose because the transformed observations must admit a physical trajectory through the allowed cortical entry disk and angular interval. After pose/warp mapping, the fitted trajectory remains constrained to the same feasible set; it is not an unconstrained regression followed by a warning.

The solver must not multiply runtime through an unconstrained exhaustive search. Feasible full-plane pose ranges are precomputed or pruned, then combined with image-derived energies. A hard surgical constraint cannot be exceeded by rounding tolerance.

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
- combinations of full-plane domain/ordered-offset/shared-normal/surgical
  constraints, plus separately labelled legacy coronal AP/order/common-tilt
  combinations;
- hard-satisfaction rate, required to be 100%;
- false infeasibility and false feasibility rates;
- result invariance when constraints are disabled;
- effect of incorrect but feasible user information.

The number and type of user-provided constraints are disclosed for every assisted comparison.
