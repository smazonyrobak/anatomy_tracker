# 141: top-four keys pass a geometry ceiling, not a learned-match gate

The predeclared **geometric** gate passed. On the same oracle truth-best original branch from the frozen 132 blind-16 beam, the v4 full-intensity head's top-one local 3×5×5 neighbourhood contained a supported key within 0.5 mm for 5,184/6,094 sites where the entire atlas lattice had such a key. The union of its top four neighbourhoods covered 5,983/6,094: **+13.11 percentage points**, above the required +10, and **98.18%** of the global lattice ceiling, above the required 60%. This is a ceiling using synthetic truth to *measure* key availability; no top-four matcher was trained or deployed.

The original scored fine matcher still hit only 938/6,094 globally accessible ≤0.5-mm v4 sites, reproduced exactly from experiment 140. It scores only within the top-one neighbourhood. Thus even where that neighbourhood contains a precise key, the head usually does not select it. The geometry gain is real but smaller than the within-neighbourhood scoring gap. Atlas-intensity and support-only arms had nearly identical key availability and scored hits; the test does **not** show that tissue anatomy rather than the atlas outline drives correspondence.

## Frozen comparison

The 132 step-2,000 parent, both 140 step-4,000 heads, atlas, 248 v4 and 243 v3 eligible DEV sections, and each section's verified original blind-16 truth-best branch ID were unchanged. Only branches with ≤1.5-mm original rigid error were examined: 179 v4 and 197 v3 sections. Neither the model nor the branch selector received the synthetic truth location. At every surviving fixed tissue query, the evaluator compared the true CCF point with supported keys in the whole 7×32×32 atlas lattice, in the top-one 3×5×5 neighbourhood, and in the top-four union. Support meant rendered atlas support >0.5. The 140 scored-hit definition was reproduced separately from an additional support-qualified version.

| DEV cohort / head | Near sections | Global ≤0.5-mm keys | Top-one local | Top-four union | Top-four gain / global | Scored fine hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 / atlas intensity | 179 | 6,094 | 5,184 | 5,983 | +13.11 pp | 938 |
| v4 / support only | 179 | 6,094 | 5,173 | 5,978 | +13.21 pp | 907 |
| v3 / atlas intensity | 197 | 6,961 | 6,118 | 6,857 | +10.62 pp | 1,146 |
| v3 / support only | 197 | 6,961 | 6,057 | 6,879 | +11.81 pp | 1,118 |

All 33,996 v4 and 38,600 v3 top-four coarse seeds in the full-intensity arm had atlas support; the support-only arm had the same counts. At 0.75 mm, the v4 full arm covered 7,474/8,445 globally reachable sites with top one and 8,343/8,445 with top four; at 1.5 mm, 8,251/8,499 and 8,479/8,499. The 0.5-mm top-four global-ceiling fractions by nearest-cardinal angle were 97.3% (<15°, 13 near sections), 98.1% (15–30°, 63), 98.7% (30–45°, 78) and 97.4% (≥45°, 25). Those are synthetic geometric availabilities, not angle-specific end-to-end pose accuracies. Top-four windows can overlap or contain repeated clamped keys; duplicates are harmless for this nearest-distance ceiling but must be removed before any scored softmax.

The frozen runner completed 752 rows for 376 distinct near physical sections, each in both arms. Post-exit audit matched config, protocol, evaluator, rows and summary SHA-256 values, verified all 752 unique cohort/section/arm keys and all 376 paired sections, rechecked per-row top-one ≤ top-four ≤ global counts, and reproduced the full arm's exact 140 fine hits. The paired panel represents only eight synthetic deformation plans/animal IDs reused across distinct planes; it is not an independent biological validation set. Output: `I:/AnatomyTracker/runs/topk_correspondence_ceiling_141`.

## Decision

The pass justifies a *bounded* scored top-four experiment, provided it directly attacks fine-key discrimination and uses matched fresh full-intensity/support-only draws plus source- and atlas-content ablations at fixed support. Simply widening the candidate union, scaling exposures, or feeding this score back into the global pose model would be unjustified: 140 blind-fit failed and 141 shows a much larger gap between available and selected fine keys. Any trained successor must be checked on blind branches and model-selected branches, not just this truth-best diagnostic. No GUI promotion, calibrated probability statement, physical arbitrary-plane claim, or public DeepSlice benchmark follows from 141.
