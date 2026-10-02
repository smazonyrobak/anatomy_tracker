# Geometric consensus 031 — frozen development result

The truth-blind consensus rule **failed** its preregistered gate on all 185 fixed synthetic DEV sections (eight identity-equal deformation plans). It chose a whole-section affine atlas plane from 030 top-16 patch lists by position/orientation agreement, with no synthetic coordinates consulted during selection. Independent replay of every saved affine against the valid tissue pixels passed its output hashes and reproduced the reported error.

| Endpoint | Result |
| --- | ---: |
| Selected rigid tissue-coordinate mean error | **3.668 mm** |
| Sections below 0.75 mm / 1 mm | 12.24% / 17.92% |
| Median query supports for selected plane | 8 |
| Fraction of selected supports that are actually within 0.5 mm of truth | **8.73%** |
| Best of 512 single-match seeds, oracle-selected by query-point truth | 0.937 mm mean query-point error |
| Truth-selected top-16 matches fitted to a plane, when ≥3 exist | **0.621 mm** mean full-tissue error on 74.05% of sections |

The last two rows are **oracle diagnostics**, not inference results. They show that useful matches and often a useful plane are in the retrieved lists, yet simple geometric agreement promotes mutually consistent false matches. The 030 pass was candidate *availability*, not usable localization. This is also why a median of eight agreeing supports is not reassuring here. The rigid endpoint is not numerically identical to 019's warped selected-mapping endpoint, but both are much too large for reliable electrode-region assignment.

This narrows the next model change: learn to identify coherent true correspondence groups and evaluate anatomical fit after rendering the implied atlas plane. Train that selector on fresh synthetic examples with difficult retrieved wrong-plane candidates and connect its pose estimate to the same model's local fitting loss. Do not spend more training on a scalar score over 019's poor fixed proposals, and do not substitute RANSAC-style consensus alone for anatomical evidence. The correspondence/pose/fitting system still needs automatic image-derived query selection, real donor development, animal-separated calibration, GUI integration and later public comparison. The raw runs are `I:/AnatomyTracker/runs/atlas_geometry_consensus_031` and `_audit`.
