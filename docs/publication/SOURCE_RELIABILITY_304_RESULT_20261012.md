# 304 — practical source-only atlas score did not pass

The repaired, predeclared run `I:/AnatomyTracker/reports/source_reliability_304_002` completed on 128 independently drawn eligible v9 TRAIN-inner sections (16 each from synthetic bases 56–63). These are new sections from previously used base animals, **not** animal-independent validation. A read-only independent audit checked the receipt, source/protocol/parent/output SHA-256 bindings, all 139 draw records, 128 distinct physical-section IDs, eight per-base counts, raw pair wins and selected-error means. The completed receipt SHA-256 is `52bd1db089c0391613c630541018dc658c2092cf8727bf2cba188a2689a5dc82`. Run `304_001` stopped on its first case due to an omitted pooling stride; it yielded no scientific result and remains preserved.

| Score on the same frozen top-16 branches | Support-matched near/far wins /108 | Mean selected rigid-site error |
|---|---:|---:|
| Frozen image-prior first choice | — | 2.286 mm |
| Source-only local-texture reliability + atlas intensity | 68 (63.0%) | 2.229 mm |
| Matched support-only score | 61 (56.5%) | 2.228 mm |
| Uniform-weight atlas intensity | 69 (63.9%) | 2.318 mm |
| Best of top 16 by unavailable truth | — | 1.048 mm |

The full score improves the frozen prior by only **0.057 mm** and is **0.001 mm worse** than support-only selection. Its seven additional near/far wins are 6.5 percentage points, below the preregistered ten-point requirement; 68/108 is also below the required 75%. No-brush near/far wins are 48/77 (62.3%, required 70%), with mean error 2.152 mm versus 2.173 mm prior and 2.111 mm support-only. Only five of eight base means improve. All gate clauses except the pair-count clause fail. Thirteen sections have fitted obliquity ≥45°; their full-score mean 2.193 mm worsens the 2.071-mm prior, but this stratum is too small for a general conclusion. The uniform-weight score is a descriptive ablation on the same cases, not an independent test of the MAD reliability rule.

The source-derived local-texture reliability is therefore **not a deployable pose score** and should not be added unchanged to the GUI or a new ranker. In contrast, 302's stronger atlas-correlation signal used a perfect tissue mask unavailable at inference, on a different reused panel. The difference focuses the next architectural work on candidate-independent tissue reliability, spatially resolved source/atlas residuals, and bounded pose/deformation feedback trained together from random initialization—not on tuning this fixed score on the same 128 cases. Preserve full-sphere training but report failures by obliquity and visible-tissue fraction; synthetic all-angle sampling does not establish physical steep-oblique accuracy. No DEV/CAL/FINAL animals, public DeepSlice benchmark, or calibrated probabilities were used.
