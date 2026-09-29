# Direct FP64 Torch mapping of an accepted subject plan

`training/arbitrary_plane_subject_torch_v6.py` contains two scientific
functions, with no change to the existing NumPy adapter or default generator.
This is an opt-in target-preparation primitive, not a model or training change.
Only syntax/source inspection precedes this commit; **GPU numerical equivalence
and acceleration have not been verified**. Do not use it to replace frozen
target coordinates before the planned bounded comparison on the actual plan.

Authenticate the accepted NumPy plan once outside these functions. Convert the
following `plan["state"]` entries once to FP64 tensors on the query device and
retain them across sections, in the mapper's positional order:

- `accepted_coarse_coefficients_um`, `coarse_origin_um`, `coarse_spacing_um`;
- `accepted_fine_coefficients_um`, `fine_origin_um`, `fine_spacing_um`;
- `global_scale`, `frozen_center_um`.

Pass FP64 `[...,3]` physical AP/DV/ML queries, the recorded
`plan["resolved_config"]["flow"]["steps"]` (currently 8), and
`identity=(plan["resolved_config"]["deformation_stratum"] == "identity")`.
`inverse=True` maps subject to CCF; `False` maps CCF to subject. The functions
do not transfer or convert plan arrays, reweight coefficients, resample a plan,
or authenticate caller input. The point mapper disables autograd and provides
coordinates only, not Jacobians or a differentiable training contract.

The field uses the original cardinal cubic basis and 64 tensor-product knot
neighbors. Lattice coordinates are `(point-origin)/spacing`, without an atlas
half-voxel offset. Safe index clamping is followed by a validity mask: missing
knots contribute zero, with no boundary-weight normalization or query clamping.
Accepted coarse and fine coefficients already include their weights/projection
and therefore sum directly. No trilinear displacement cache is substituted.

Forward mapping applies the recorded RK4 positive-velocity flow, then positive
diagonal scale about the frozen centre. Inverse mapping removes that scale
first, then applies the negative-velocity RK4 flow. Four stages per integration
step and their original arithmetic ordering are retained. This preserves the
finite-step algorithm, not an algebraically exact inverse of its discrete
forward map. Query tiles default to 8192 points; both fields remain resident.

The vectorized 64-neighbor reduction changes summation ordering, so even FP64
does not imply bitwise NumPy equality. GPU fusion/rounding and nearby annotation
boundaries require a bounded comparison of actual frozen-plan mappings before
acceptance. A new evaluator's source/hash must remain explicit; neither its
outputs nor receipts inherit old bytewise identities automatically. No GPU
work is authorized by this implementation commit while rehearsal is active.
