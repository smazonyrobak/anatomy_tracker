# Data provenance and synthetic generator

## Source data

### Allen CCFv3 25 µm

`average_template_25.nrrd` supplies the anatomical intensity volume and `annotation_25.nrrd` supplies brain support and Allen structure identifiers. Atlas files, ontology/query table and bregma conversion constants are checksum-pinned. The CCF supports exact synthetic plane, mask and label supervision; it does not represent every individual's anatomy.

### Allen registered histology

The frozen AtlasPose baseline uses Allen Product 5 connectivity serial two-photon/block-face sections as trusted real-image pose supervision and checkpoint-selection data. Product 8 slide-mounted affine labels are diagnostic only because specimen-level offsets were observed. Product 8 cannot enter training, selection, calibration or a release gate unless a future, independently documented curation changes its role.

Product 5 contributes realistic appearance and registered plane labels. It does not automatically provide exact individual-anatomy dense deformation ground truth.

### DeepSlice public benchmark

The published human-aligned dataset, the 1,400-section local comparison cohort and their prior predictions are comparator evidence only. They have already been inspected during this project's development and are not a pristine hidden test. Their original development/test brain assignment and raw bytes/ground truth are preserved. Audit invalidated the prior local DeepSlice comparison because the raw raster view did not match DeepSlice's intended A-to-P input convention. The corrected reproduction uses the single hash-bound horizontal image-frame adapter defined in the coordinate contract; it never reflects the physical atlas mediolateral axis.

### Local real sessions

Session 722 and any previously opened local images are development/diagnostic cohorts. They may test workflow and reveal failure modes but cannot support an untouched generalization claim. New real benchmark animals must be independently collected and split before model development.

## Unit of splitting

Animal is the mandatory split and inferential unit for real-data validation.
Every experiment, serial section, repeated acquisition channel, thumbnail, crop
and augmented descendant from one animal inherits that animal's split. A real
record without a resolvable animal identifier is ineligible for confirmatory
claims; experiment identifiers remain required provenance but never substitute
for animal grouping. Hash-based audits reject duplicate or near-duplicate
images across splits. Laboratory-held-out and acquisition-held-out subsets are
separately identified.

Synthetic data use the same grouping rule. Every view from one sampled 3-D
anatomical warp receives one immutable `synthetic_animal_id`; all blocks,
sections, channels, crops, brush descendants and augmentations from that warp
remain in one split. A `synthetic_block_id`, monotonic `section_index`, physical
section spacing and shared block frame are retained so sequence-aware methods
can be evaluated without allowing related slices to leak across splits.

## Hierarchical pseudo-animal geometry

The release-eligible generator uses this causal order:

1. sample one low-frequency 3-D diffeomorphism for a pseudo-animal;
2. define a flat arbitrary physical plane and finite optical support in that
   pseudo-animal's space;
3. map all axial/raster samples through the inverse animal transform into Allen
   coordinates and sample template and integer annotations there;
4. save the exact atlas-coordinate map, its best-fit finite O/U/V plane and the
   residual 3-D deformation about that plane;
5. apply section-specific affine-free 2-D processing deformation;
6. apply non-invertible damage geometry, then crop/window;
7. synthesize modality/appearance/background, optional smart-brush descendants
   and the final explicit raster reflection.

This ordering is based on the generative diffeomorphic mouse-brain framework of
[Tward et al.](https://doi.org/10.1038/s41467-025-65317-7). A physical plane in
an individual brain generally maps to a mildly curved surface in atlas space;
adding only a 2-D warp after slicing the atlas would conflate biological shape
with slide processing. The implementation may evaluate the shared 3-D field at
requested section points rather than materializing a warped full-resolution
volume, but the exact field identity and sampled coordinate maps remain bound.

The v2 centre/slab implementation remains an identity-anatomy acquisition and
quadrature precursor. A separate standalone animal stage now samples
animal-scoped, coarse-plus-fine cubic B-spline stationary velocity fields from
random initialization, removes the complete local affine component, keeps only
a positive diagonal global anatomy scale, and separates
`subject_deformation_plan_id`, `subject_deformation_realization_id` and
`synthetic_animal_id`. Its accepted realization is bound to fixed forward and
inverse maps, Jacobians, cycle errors, final float32 affine residuals,
conservative continuous-field bounds, exact replay and an authoritative CCF
domain check. A coefficient-derived global gradient bound also certifies every
fixed-step RK4 update: for `q = ||Dv||_F / N`, accepted candidates require
`q + q^2/2 + q^3/6 + q^4/24 < 1`, which keeps each forward and inverse step
orientation-preserving between audit nodes. It has no learned, pretrained or
previous-model dependency.

This is still the shared-animal stage, not the complete pseudo-animal
generator: the finite-depth coordinate pullback, section-processing warp,
damage, appearance and brush descendants remain downstream, and only that
complete chain may issue the final `synthetic_realization_id`. The frozen
half-fine-knot audit grid contains 83,754 points for representative Allen bounds
at 500-um fine spacing; the current CPU implementation exceeded 180 seconds
even for an identity timing probe. The density contract is retained, but the
audit must be vectorized or moved to GPU before production-scale generation.

Initial animal-warp bounds are explicit engineering priors, not claimed
population estimates: global scale `0.80--1.25`, low-frequency local Jacobian
determinant approximately `0.5--2`, and a separate stress stratum outside the
ordinary envelope. Individual variation often exceeded processing effects in
the Tward cohort, while local length-scale changes approached `1.25x`; exact
training bounds remain versioned and must later be checked against real
histology.

## Unified synthetic sample contract

Each generated sample records:

- generator/version and manifest hashes;
- source atlas/data hashes;
- `synthetic_animal_id`, block/sequence ID, section index and all hierarchical
  seeds;
- subject-space plane/frame, animal-level 3-D warp and exact inverse
  atlas-coordinate raster;
- constrained full 3-D centre/frame/basis pose and exact QuickNII O/U/V, with
  AP/L--R/D--V only as a derived coronal compatibility view;
- in-plane rotation, scale and translation;
- fixed template, fixed labels and brain mask;
- moving grayscale section and pre-artifact image;
- exact forward/inverse maps and stationary velocity when applicable;
- animal/section Jacobian summaries plus valid-correspondence, fold, visible,
  damaged and missing-tissue masks;
- model-input outline, outline-availability flag, outline perturbation and masking mode;
- artifact types, parameters and severity;
- positive plane and prespecified wrong-plane candidates.

Generation is on demand from immutable manifests so 500,000 training views need not become 500,000 persistent image files.

## Geometry distribution

The frozen v2 AP/tilt generator remains unchanged for provenance. New
release-eligible work uses a versioned v3 arbitrary-plane contract:

- sample a Gaussian direction and normalize it to obtain a surface-area-uniform
  normal on the antipodally identified sphere, not uniform Euler angles;
- sample in-plane roll uniformly over a full turn;
- sample signed offset uniformly over the annotation foreground projected onto
  that normal, then verify actual rendered support;
- retain named near-tangent, tiny-support and anatomically ambiguous stress
  strata without silently changing the reference validation measure;
- store the constrained centre/frame/positive-triangular-basis state and exact QuickNII O/U/V,
  projection bounds, offset quantile, tissue support, RNG stream, rejection
  attempt, atlas hashes and synthetic-realization identity;
- bind raster width and height because QuickNII pixels use `x/W,y/H` and a
  discrete flip shifts `O` by `(W-1)/W U` or `(H-1)/H V`; no new synthetic
  renderer may use the frozen diagnostic's `+0.5` sampling rule;
- use nullable animal/specimen IDs for atlas-derived samples and preserve exact
  animal/specimen/experiment IDs for all real descendants;
- image scale from 0.5 to 1.5;
- bounded translations;
- radial, anisotropic stretch and swirl velocity components;
- local expansions/compressions, asymmetric warps and compound deformations;
- positive-Jacobian forward and inverse ground truth;
- tears, missing cortex and occlusions represented as invalid tissue rather than deformation.

The plane-only sampler uses `plane_realization_id`. The name
`synthetic_realization_id` is reserved for the later complete record that binds
the frame, in-plane basis, QuickNII O/U/V, deformation, appearance, mask and
rendered artifacts; it is intentionally absent from plane-only manifests.

The first v3 manifest implementation is an exact small-volume geometry
prototype. It materializes occupied voxel centres to test the reference
measure, intersection rule, replay hashes and provenance without touching the
frozen v2 path. That representation is deliberately not approved for full
Allen-scale generation. A separate hash-bound compact-support backend replaces
its linear per-plane occupied-voxel scan at Allen scale.

The implemented development-only full-scale backend uses connectivity plus a
compact convex-support index. For each line along the AP array axis it retains the first
and last occupied voxel centre, computes a lexicographically canonical convex
hull of those integer endpoints, and obtains every directional projection
bound from that hull plus the anisotropic voxel-box half extent. The pinned
Allen annotation has 32,387,385 foreground voxels in one 6-connected component;
the hash-bound preflight in
[`../../publication/arbitrary_plane_geometry_preflight.yaml`](../../publication/arbitrary_plane_geometry_preflight.yaml)
reduced 206,635 unique scan-line endpoints to 3,194 hull vertices. Because the
closed occupied-voxel union is connected, its projection
onto any normal is one interval, so a plane offset lies inside those bounds if
and only if the plane intersects support. This removes both the multi-gigabyte
point cloud and per-plane full-mask scan. The stored integer endpoint audit set,
hull, source and Qhull/dependency identity are hash-bound; analytical and
seeded small-mask tests plus the pinned-CCF endpoint parity audit define the numerical tolerance.
Disconnected masks retain separate component intervals rather than using the
single-interval fast path.

The support backend is now integrated into a development-only finite-render
precursor. It samples deterministic RP2 normals, roll and offsets from the
merged component-interval union; uses isotropic physical pixel pitch with
symmetric padding; emits official QuickNII `x/W,y/H` geometry; and renders CCF
intensity, uint32 annotation IDs and tissue masks. A prepared immutable context
hashes the full Allen assets once and supports random-access samples without
per-sample atlas rescans. Separate `plane_realization_id`,
`finite_plane_geometry_sha256`, `rendered_artifacts_sha256` and
`finite_plane_render_id` layers prevent raster/template choices from being
mistaken for plane identity. The exact effective float32 renderer grid and its
Allen, physical and QuickNII O/U/V are bound alongside the float64 design
geometry. The small pinned-Allen preflight in
[`../../publication/arbitrary_plane_rendered_preflight.yaml`](../../publication/arbitrary_plane_rendered_preflight.yaml)
passed near-cardinal and extreme-oblique samples plus exact cached replay. It
is a geometry/render feasibility result, not model or accuracy evidence.

This is still not a complete rendered synthetic realization: deformation,
appearance, damage, moving images and accurate/imperfect/absent outline modes
remain unbound. Consequently `synthetic_realization_id` is still withheld and
the large training manifest remains unfrozen.

Future finite-raster wrong-plane candidates use geodesic normal perturbations,
physical normal-offset perturbations and coupled anatomically confusing planes.
Every positive and candidate must intersect the brain; their exact O/U/V and
physical candidate distances must be hash-bound. Pose ranking must remain
sensitive after an optimal bounded local warp.

The current candidate prototype remains an unoriented infinite-plane proposal.
The positive finite-render precursor now binds roll, centre, basis, O/U/V and
reflection state, but wrong-plane candidates do not yet inherit that finite
geometry. Before candidate images become training-eligible, the verified base
frame is minimally rotated/parallel-transported to each candidate normal, an
explicit roll perturbation is applied, and every candidate raster is required
to contain a prespecified amount of rendered tissue.

## Appearance distribution

All model inputs are grayscale. Include clean through severe combinations of:

- gain, offset, gamma, polarity and nonlinear tone curves;
- black through grey backgrounds and unequal hemisphere illumination;
- local over/underexposure and clipped blowouts;
- tiling seams, tile gain variation and within-tile vignette;
- noise, blur, streaks, thin scratches and bright specks;
- bubbles, tears, edge loss, polygon occlusion and edge-to-edge blackout;
- label-conditioned synthetic appearance to break dependence on CCF template intensity.

Appearance is modality-conditioned rather than one universal grayscale
perturbation. Brightfield integrates stain in optical-density space before
Beer--Lambert conversion. Fluorescence uses explicit photon gain, background,
PSF/blur and Poisson--Gaussian noise. Confocal/two-photon optical planes remain
distinct from physical cut thickness, while an unknown-mode mixture covers
single-focus, extended-focus and projected exports. Parameters are stored even
when the final model input is grayscale.

The proposed initial mixture is 10% clean, 45% mild, 35% moderate and 10% severe. Exact frequencies are frozen after a blinded visual generator audit and before large-scale training.

The literature does not establish universal mouse-histology artifact
frequencies. These proportions and the initial deformation/damage magnitudes
are provisional engineering priors, must be named as such in manifests, and
are replaced or sensitivity-tested after descriptive measurements on eligible
real training/development histology.

Tears and missing tissue never receive dense-correspondence targets. Folds are
stored as a separate potentially multi-valued visibility event, with the
visible top-layer coordinate map if available. Detached pieces retain their
own rigid transform and validity. These masks are independent of the presented
smart-brush outline.

## Outline and background curriculum

The deployment model must not require a manually painted outline. Each source
view is assigned one of three hash-bound input modes: accurate outline with a
black exterior, realistically imperfect outline with a black exterior, or no
outline with the acquired/synthetic background retained. An explicit
availability input tells the model which contract applies. Outline errors are
generated independently of tissue damage and dense-flow validity; they never
alter the exact anatomical correspondence targets.

Validation retains the same underlying cases across three separately named
tracks: automatic with no user outline, a common frozen automatic-outline
sensitivity analysis, and smart-brush-assisted registration. The public
automatic comparison cannot silently use a hand-corrected mask.

## Generator QA

Before training:

- inspect stratified montages with transform metadata hidden from the reviewer;
- numerically compose forward/inverse maps;
- check label resampling and validity masks;
- require positive Jacobians for synthetic ground-truth deformation;
- verify pose recovery from rendered plane anchors;
- check every corruption both alone and in realistic combinations;
- ensure the model cannot infer pose from canvas scale, padding, filenames, random seeds or artifact parameters.

## Independent synthetic test generator

The locked exact benchmark uses an independently implemented transform family, different seeds and disjoint manifests. Where training uses the repository's velocity components, testing should add independent B-spline/TPS-like or alternative diffeomorphic fields, independently generated tears/backgrounds and separate artifact textures. Results must state that all anatomy still derives from CCF and therefore cannot replace real external validation.

## Data cards and amendments

Every dataset receives source, license/terms, acquisition, animal count, section count, exclusions, intended role, split, hashes and known limitations in [`../../DATA_CARD.md`](../../DATA_CARD.md) or a linked immutable manifest. Any protocol amendment is dated, justified and marked as occurring before or after access to the affected benchmark.
