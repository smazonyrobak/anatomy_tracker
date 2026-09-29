# Coherent-subject plan cohort 002 — frozen protocol and completed result

`training/prepare_joint_v6_coherent_subject_plans.py` prepares **8 training and 4 development** independently seeded, accepted 3D subject maps. Root seed is **2026092911**; future output is `I:/AnatomyTracker/data/joint_v6_coherent_subject_plans_002`. This source-only protocol does not establish that any of these plans has been generated or accepted.

The preparation-time protocol below is preserved. Execution subsequently completed;
the result and exact completion binding are recorded at the end.

## Subject identity and split

Training indices are 0–7; development indices 0–3. The existing v2 seed derivation includes the split, so equal numeric indices across splits do not share stochastic fields. Each has a persistent `joint-v6-coherent-cohort-002-{split}-{subject|animal|specimen|experiment}-{index:08d}` ID, plus the sampler's content-bound `synthetic_animal_id`, plan ID and realization ID. Lineage is saved separately without modifying the strict existing plan representation.

All descendants of one map stay in its preassigned split. These are distinct synthetic anatomies constructed from **one Allen atlas**, not independent biological animals or evidence of biological generalization. There is no reuse of the single pilot subject, old model weights, learned features, or pseudolabels.

## Frozen generation rule

Reuse the existing `sample_animal_subject_deformation_plan_v2` **standard** defaults: positive diagonal global scale about the full CCF centre; affine-free coarse/fine cubic B-spline stationary velocity fields; fixed RK4 integration and existing numerical acceptance gates. Fields are drawn once per subject from domain-separated PCG64DXSM seeds. The fixed candidate amplitudes are 125, 62.5, 31.25, 15.625, 7.8125 µm, accepting the first passing candidate. There is no subject-seed redraw, identity fallback, or gate relaxation; failure propagates and leaves completed earlier plans intact. The complete candidate history is retained. This samples the existing accepted synthetic deformation distribution, not a fitted distribution of biological variation.

The loader verifies pinned raw Allen template/annotation files and decoder, then requires the established v2 context hash `c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8`. Coordinates are AP/DV/ML µm, voxel-face origin 0 and 25 µm spacing; full physical bounds come from the actual scalar volume shape. The context's receipt, atlas input hashes, runtime/source provenance and bounds are saved. Atlas arrays are then released; the CPU-only sampler needs those bounds and the context identity, not repeated atlas decoding.

## Artifacts and execution boundary

Each `train/subject_XXXXXXXX` or `development/subject_XXXXXXXX` directory contains the existing `subject_plan.metadata.json` + `subject_plan.arrays.npz`, the recomputed content receipt, lineage, and completion record with exact file hashes, accepted amplitude/scale and candidate audits. The root records the full predeclared roster, context, archived source/preflight files, exact source hashes and Git HEAD. It emits a root completion inventory only after all 12 plans finish and source bytes remain unchanged. Output creation is fresh-only; no overwrite or automatic resume is implemented. A failed run is not a complete cohort.

Stdout reports each subject start and accepted/frozen milestone. Content authentication is not a redundant full independent numerical replay. No GPU, model inference, sections, appearance synthesis, masks, or benchmark are used by this script.

**Later, separately implemented stage:** after the GPU coordinate check, plan 64 planes per training subject (512 total) and 32 per development subject (128 total). Reuse each exact frozen 3D map across its sections, with independent section/appearance seeds. Preserve conservative full brain-intersecting-plane coverage and censored marginal draws; derive masks/eligibility from newly mapped anatomy. Those section-generation decisions and outputs are not implemented here. Keep all held-out subject descendants out of training; a 12-subject one-atlas engineering cohort cannot calibrate biological electrode-site probabilities.

Future command from the repository (not run during preparation):

```powershell
& 'I:\AtlasJointProject\envs\npixel_analysis\python.exe' -m training.prepare_joint_v6_coherent_subject_plans
```

## Completed result — 2026-09-29

Terminal9774 exited0 after3117.55s, source `673b8fcae0a06fd549b9d238b8789e04f68f2bcc`.
All12 predeclared subjects completed, with12 unique animal, specimen, experiment
IDs and content-bound plan receipts. After confirmed exit, all71 files in the
root completion inventory were independently rehashed and matched. Completion
SHA-256: `83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89`.

Every subject accepted candidate1 at62.5um after the fixed125um candidate failed
the interpolation-halo gate (two also failed the speed bound). Every accepted
candidate's stored failed-gate list is empty. Across accepted numerical audit
grids, minimum composed Jacobian determinants range0.7830–0.8631; maximum local
displacements range141.15–181.27um; maximum forward-then-inverse cycle discrepancies
are approximately0.71–1.49e-9um. These are authenticated generator diagnostics,
not an independent rerun of integration, model performance or biological evidence.

The separate section generator is now bound to this exact completion. Its
predeclared512 training+128 development planes and three paired appearance modes
are unchanged. This plan stage generated no sections or trained weights.
