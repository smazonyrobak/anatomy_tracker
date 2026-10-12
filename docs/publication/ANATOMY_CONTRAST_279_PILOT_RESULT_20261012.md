# 279: anatomy-conditioned synthetic contrast helps a weak real orientation readout, but fails the progression gate

The predeclared 1,000-batch comparison completed. Both arms continued the same scratch-lineage 275 step-8,000 pose checkpoint and optimizer state, each with 4,000 independently sampled brain-intersecting TRAIN synthetic sections. The only training contrast was the original v9 appearance versus v11's fresh, per-plane annotation-conditioned tissue contrast and texture. No acquired affine or image supplied a gradient. All 98 acquired sections below are median sections of previously inspected TRAIN donors, not an independent biological test; their inherited Allen affines are weak orientation references, not physical pose truth.

| Fixed batch-1,000 readout | v9 control | v11 treatment |
| --- | ---: | ---: |
| New 275 synthetic mean first-choice site error | 2.191 mm | 2.052 mm |
| New 275 synthetic top-16 capture within 1.5 mm | 61/64 | 58/64 |
| Frozen 270 synthetic mean first-choice site error | 2.464 mm | 2.511 mm |
| Acquired coronal first-choice plane family | 58/58 | 58/58 |
| Acquired sagittal first-choice plane family | 24/40 | 30/40 |
| Acquired sagittal median weak normal-angle discrepancy | 45.74° | 27.02° |

The **conjunctive gate failed**: the first-choice synthetic improvement was 0.139 mm, below the specified 0.20 mm, and top-16 capture lost three cases, above the allowed two. The older synthetic-panel regression was 0.047 mm (allowed ≤0.10); sagittal family agreement gained six cases, three in each donor-hash parity stratum, and its weak angle fell 18.72°; coronal agreement lost none. Those passing clauses do not override the two failed clauses. At batch 500, sagittal agreement was already 30/40 for control and 31/40 for treatment, so the fixed batch-1,000 difference should not be portrayed as a stable six-case treatment effect.

The three new-panel top-16 losses all changed from a control hit to a treatment miss, although their individual first-choice errors improved; the treatment improved first-choice error in 42/64 cases and worsened it in 22/64. These descriptive observations do not change the fixed gate. The acquired orientation result is a useful transfer signal, but neither it nor the synthetic texture census establishes real slice-position accuracy, local deformation quality, or physical steep-oblique performance. The generator's brightness-tail and legacy imperfect-brush square-void limitations remain.

The independent verifier checked the frozen protocol, source/input/output and checkpoint hashes; 8,798 attempted draws, 4,000 accepted sections and 1,000 batches per arm; 24 row-level readout blocks; 48 fresh image-to-pose prediction replays; and exact gate arithmetic. It passed the audit while confirming `gate_pass=false`. Frozen output: `I:/AnatomyTracker/runs/anatomy_contrast_279_pilot_001`. The fixed protocol is [ANATOMY_CONTRAST_279_PILOT_PROTOCOL_20261012.md](ANATOMY_CONTRAST_279_PILOT_PROTOCOL_20261012.md).

**Decision:** do not scale this exact v11 continuation or promote its checkpoint. The next bounded work should test, on fresh brain-intersecting TRAIN sections, whether anatomical fit difficulty actually distinguishes *predicted* near-versus-wrong planes and improves their first-choice order while preserving map quality; prior joint-feedback experiments failed this decisive blind-ranking test. Only then is a jointly optimized pose-and-local-map training run justified. Independent donor-disjoint real validation, calibrated electrode probabilities, GUI replacement, and a fair DeepSlice comparison remain open.
