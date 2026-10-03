# Retrieval-seed pool 034 — frozen read-only diagnostic

Using the unchanged 031 truth-blind geometry score, rank all 512 atlas-match seeds per DEV section, then use synthetic truth **only to measure** the minimum rigid 32-query-coordinate error in each pool. The pooled minimum is an oracle upper bound for any later score/fitter over that pool, not a selected prediction. All 185 sections/eight synthetic identities were included; no weights changed, and no image was rendered or viewed.

| Top-ranked seeds retained | Oracle-best query error | Sections with a seed <0.75 mm | <1 mm |
| ---: | ---: | ---: | ---: |
| 1 | 3.703 mm | 6.58% | 15.79% |
| 8 | 2.331 mm | 15.44% | 29.86% |
| 16 | 1.836 mm | 20.45% | 39.29% |
| 32 | 1.488 mm | 25.97% | 46.55% |
| 64 | 1.280 mm | 26.49% | 51.65% |
| 128 | 1.138 mm | 27.45% | 56.33% |
| 512 | 0.941 mm | 31.89% | 62.28% |

The geometry rank loses many physically useful seeds. This query-point endpoint also excludes subsequent local warp and is not directly the same as 019's full-tissue rigid metric. The result weakens the case for a large atlas-template bank as the centre of the final system: it is costly in memory and does not supply reliable near-truth top-ranked proposals on its own. The atlas patch encoder remains useful for differentiable image-to-render evidence around direct pose candidates. Frozen output: `I:/AnatomyTracker/runs/atlas_consensus_pool_034`.
