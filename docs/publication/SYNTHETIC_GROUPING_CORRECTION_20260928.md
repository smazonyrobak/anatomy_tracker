# Synthetic grouping correction and scalable subject variation

The substantive cache's 320 training and 40 development "synthetic animals"
are **organizational groups**, not independently generated consistent 3D
anatomies. The 5,120/640 section counts, empty identity intersections and frozen
receipts remain correct. Saved `animal_macro` statistics must be interpreted as
`synthetic_group_macro`: separate atlas-rendering groups, not biological units.
More views or more arbitrary ID strings do not create new biological subjects.

## Exact source evidence

- `run_arbitrary_plane_substantive_finite_data_v6.py` sets 16 sections per animal
  and builds 3,072/384 pose rows plus 2,048/256 joint rows using the finite-v4/v5
  curriculum paths and a single pinned Allen template/annotation.
- `arbitrary_plane_finite_pose_curriculum_v4.py:910` and
  `arbitrary_plane_finite_joint_curriculum_v5.py:1193` derive
  `animal_index = sample_index // sections_per_animal`; all five lineage ID
  strings are then constructed from that index or the section sample index.
- The pose `_derived_seed` at line 74 hashes the root seed, sample index and
  domain. The joint equivalent at line 121 additionally includes attempt index.
  Plane, finite thickness and synthetic realization streams are thus row-based.
- `arbitrary_plane_synthetic_generator.py:235` derives G1 deformation, G2
  appearance and G3 damage/outline RNG streams from root seed, split,
  **sample_index**, stage, field and attempt. `_rng` at line 261 uses that stream.
  The arbitrary animal label is provenance, not a shared subject latent variable.
- These curriculum imports use the section-level synthetic generator, not
  `sample_animal_subject_deformation_plan_v2` or the subject-slab-v2 path.
  Pose rows force identity G1; joint rows have section-level affine-free G1.

The real Allen cohort is different: its 64 identifiers are actual upstream
`Donor.id` values, split into 58 training and six development donors with no
within-cohort ID overlap. This does not resolve historical exposure or establish
untouched final validation; the existing cohort limitations remain in force.

## Small productive expansion

Keep the current exact-catalogue clean-slab curriculum as an optimization
intervention, mixed with the frozen complex arbitrary-plane rows. Count unique
rendered observations, catalogue-cell coverage and replay exposures separately;
discrete cell coverage is not continuous subcell pose coverage. Use a monotone
sample counter and independent train/development RNG domains, saving the exact
cell/physical state, thickness, mode, seed and group arrays once per chunk.
Authenticate source volumes and frozen rows once per run, and hash whole chunks
when closed rather than serializing/replaying a receipt ledger each step.

Preserve all three input modes, continuous-plane complex rows and finite physical
slabs: nine samples over 25–100 µm with global weights
`(1,2,2,2,2,2,2,2,1)/16`, zero padding before summation, and no per-pixel in-bounds
renormalization. Low-support examples are not discarded or assigned confident
point supervision. At eight new rows per 16-row batch, 16,000 steps produce
128,000 new observations; the remaining half-batch can replay complex rows.

## Efficient actual subject variation: proposed, not implemented

Reuse `sample_animal_subject_deformation_plan_v2` once per synthetic subject and
retain its exact plan/seed/ID across that subject's sections. Its existing
default combines a stable diagonal scale (log half-range 0.03), 1,000/500-µm
coarse/fine cubic B-spline stationary velocity fields, and eight-step RK4.
The default nominal amplitude is 125 µm, with an audited deterministic amplitude
backoff. Distinct subject plans, not sections, belong wholly to one split.

Keep one original atlas resident. Reuse the plan's verified subject-to-CCF
mapper and evaluate only the `9 × 96 × 96` physical sample coordinates of each
section; then interpolate the atlas at those mapped coordinates and integrate
the PSF. This is the existing `arbitrary_plane_subject_slab_v2.py` pullback idea;
it needs no full deformed intensity volume per image. Amortize plan generation
and authentication over many sections, with independent section appearance,
damage and brush seeds. Preserve exact mapped centre/slab correspondences.

The current mapper is CPU NumPy and cubic-B-spline RK4 can dominate throughput.
First measure a small subject block. If necessary, tabulate only its coordinate
displacement field on a coarse grid once per subject (not a deformed intensity
volume), then sample it on GPU. A 100-µm grid over the Allen extent is roughly
15 MB for three float32 displacement channels, reused over hundreds of sections.
Grid interpolation changes the operator: compare it with the exact v2 mapper
and recheck physical error/cycle/Jacobian bounds before using its correspondence
labels. No equivalence or speedup is assumed without that measurement.

After nonlinear subject mapping, an acquired flat plane is generally a curved
surface in atlas space. Do not attach the original unwarped plane as exact
ground truth. Use the retained 3D pullback correspondences and an explicit
pose/deformation decomposition; an unconstrained normal displacement cannot be
represented by only a 2D in-plane warp. This integration and its identifiability
decision are required before claiming actual subject variation is learned.
