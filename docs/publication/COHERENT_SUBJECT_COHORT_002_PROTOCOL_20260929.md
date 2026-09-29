# Coherent-subject plan cohort 002 — prepared, not executed

`training/prepare_joint_v6_coherent_subject_plans.py` prepares **8 training and 4 development** independently seeded, accepted 3D subject maps. Root seed is **2026092911**; future output is `I:/AnatomyTracker/data/joint_v6_coherent_subject_plans_002`. This source-only protocol does not establish that any of these plans has been generated or accepted.

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
