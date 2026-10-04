# Frozen 083 whole-plane evidence: raw score fails

The predeclared 084 diagnostic ran to completion on the 246 eligible 061 synthetic DEV sections from eight held-out deformation plans. It used parent 059's unchanged blind 8-old/6-anchor beam and frozen 083 step-6,000 match embeddings. No weights were updated; the same single label-free confidence formula was applied to all 14 candidates. Exact synthetic coordinates were used only to measure rigid error and identify the best *existing* candidate for analysis, never to construct or score a candidate at inference.

| Plan-equal metric | Result |
| --- | ---: |
| Best existing candidate rigid error | 1.022 mm |
| Parent prior-selected rigid error | 2.656 mm |
| Raw 083 match-confidence-selected rigid error | 3.065 mm |
| Truth-best existing candidate's mean match-score rank | 6.82 / 14 |
| Truth-best in match-score top three | 24.6% |
| Truth-best ranked first | 8.0% |

The raw score worsened selected rigid error by 0.409 mm relative to the parent prior, and top-three recall missed the predeclared 50% threshold. It worsened all three background/brush strata: exact-black 2.542→3.080 mm, raw 2.775→3.174 mm, imperfect-brush 2.558→2.914 mm (prior→match-selected). Thus it fails both required conditions for using this raw match score directly. The mean truth-best score margin over the strongest other branch was negative (−0.0155). This is a candidate-ranking failure, **not** a reversal of 083's held-out local correspondence gain. A candidate-specific softmax peak is not a calibrated cross-plane probability or a substitute for comparing fitted anatomical maps.

**Next targeted step:** train an explicit score on all 14 existing blind branches using synthetic physical mapped error and deformation cost from the actual atlas-fitting path, with no true-plane insertion and no weak Allen affine as anatomical truth. Only then test whether fit-informed branch selection improves the same full-beam held-out metric; allow pose and local map updates jointly only if rank learning gives a useful signal. A negative result from this one fixed confidence aggregation does not prove a learned ranker cannot work. No real-animal accuracy, GUI promotion, calibrated uncertainty, final-test access or DeepSlice superiority is claimed.

Frozen output: `I:/AnatomyTracker/runs/dense_atlas_branch_ranking_084_diagnostic`. Its 3,444 rows equal 246 sections × 14 candidates. The independent read-back matched config SHA-256 `44563a8f7185ab358952906895968e0e2f732f1b9033d8f2624969ceaf237c78`, rows `983c0f7b0d9b9ea6954fd026acfc0e2861daa928833b6d84ba4937578267ab80`, summary `351320aa0cef9bed1bb0b9c247d3b3024fa8f4e0aa5875429929418b5c2e10d7`, and the recorded script, 083 pilot and DEV-panel receipt hashes. The metric here is rigid plane error, not the final warped tissue error.
