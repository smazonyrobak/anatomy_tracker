# Rotation-group pooling 029 — frozen result

The independent verifier passed on the same 185 held-out synthetic DEV sections and eight synthetic identities, with the 025 batch-3,000 weights frozen. No training, final-test animals or public benchmark was used. The oracle deformed atlas plane supplied a 288-patch bank with an exact positive for each of 32 query points, so this remains a component test rather than unknown-plane localization. Source/checkpoint/panel/result hashes and raw per-section rows are bound in `I:/AnatomyTracker/runs/atlas_group_pool_029`.

| Descriptor | Top-1 within 0.5 mm | Top-16 within 0.5 mm |
| --- | ---: | ---: |
| Original unpooled 025 | **96.22%** | **99.91%** |
| Average 4 quarter-turns | 83.67% | 98.49% |
| Average 4, atlas shifted halfway by 45° | 11.09% | 59.51% |
| Average 12 30° rotations | 72.66% | 97.39% |
| Average 12, atlas shifted halfway by 15° | 41.06% | 90.63% |

At 12 rotations, the unshifted raw/black/imperfect-brush top-1 values were 61.09% / 94.22% / 61.21%. The preregistered ≥80% unshifted and ≤10-point half-step-loss gate **failed**. Simple group averaging of a network trained only on aligned orientations removes too much useful structure and does not generalize smoothly between sampled angles. Together with 028's failed full-roll augmentation, this rejects two cheap substitutes for explicit orientation handling. It does **not** refute properly rotation-equivariant features.

The evidence-led next step is an explicit **hierarchical** atlas orientation search using the frozen 025 feature only where bank resolution and size are tractable, with atlas-point and angular coverage ceilings recorded. Candidate matches must then undergo geometric consensus before they can initialize the *same* joint pose/deformation model. The experiment must measure genuinely unknown-plane proposal capture and not use true centers, masks or normals at inference. Only then can joint training, donor-separated real validation, calibrated electrode probabilities, GUI integration and fair DeepSlice benchmarking become credible.
