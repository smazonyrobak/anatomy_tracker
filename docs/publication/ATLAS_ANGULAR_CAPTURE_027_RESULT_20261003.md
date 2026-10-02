# Atlas patch angular capture 027 — frozen result

The independent verifier passed on all 185 synthetic-development sections, eight disjoint synthetic identities and 4,625 condition–section rows. The batch-3,000 descriptor from 025 was frozen; no weights, final-test animals or public benchmark were used. Atlas patches were finite-thickness renders centered on the **known** correct CCF point; only orientation was perturbed. Raw results and source/checkpoint/panel hashes are in `I:/AnatomyTracker/runs/atlas_angular_capture_027`.

| Plane perturbation | Top-1 match within 0.5 mm among 32 points |
| --- | ---: |
| Correct rigid orientation | 88% |
| Normal tilt 5° | 81% |
| Normal tilt 10° | 68% |
| Normal tilt 20° | 50% |
| Normal tilt 40° | 32% |
| In-plane roll 15° | 50% |
| In-plane roll 30° | 28% |
| In-plane roll 60° | 7% |
| In-plane roll 90° | 4% |

Each nonzero row averages both signs and, for tilt, both in-plane axes; within each condition synthetic identities are weighted equally. The predeclared ≥40% tolerance at 10° tilt and 15° roll with no appearance-mode collapse **passed**. This supports a *coarse-to-fine orientation search* using the 025 descriptor, but it also shows that roll is the expensive dimension: about 30° steps already halve its correct-point retrieval, so a naive all-brain × all-normal × all-roll feature bank would be very large. [SLIV-Reg](https://arxiv.org/html/2410.18683v1) uses a rotation-equivariant encoder and many sampled planes; its same-subject CT/MRI performance still does not establish transfer to our histology/atlas task.

The next targeted change should train an independently in-plane-rotated atlas-positive descriptor from random initialization, matched to 025 in all other respects, and check that true-plane accuracy survives while roll tolerance expands. If it does, search can keep its normal-direction samples while reducing roll bins; if not, use an explicitly rotation-equivariant representation or retain a hierarchical roll search. This diagnostic alone does not locate an unknown plane, recover local tissue warp on real sections, calibrate electrode probabilities or ship the GUI model.
