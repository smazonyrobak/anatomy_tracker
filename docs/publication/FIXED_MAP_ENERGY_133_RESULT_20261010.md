# 133: fixed-map fit energy still did not identify atlas anatomy

The predeclared read-only assay finished on the frozen 132 step-2,000 checkpoint and all 256 fresh v4 synthetic DEV sections (248 informative, eight retained as ineligible). It compared the same 16 blind pose candidates and the same fitted tissue maps in every arm. The new score compared source and finite-thickness atlas descriptors **at each candidate's fitted coordinates**, with one image-derived pixel weighting shared by all candidates, plus explicit atlas-coverage and deformation costs. It had no per-pixel free search or direct CCF-coordinate input. The candidate maps were unchanged by the ablations. The known synthetic thickness was supplied to all arms, which is optimistic relative to unconstrained real use.

| First-choice rule | Plan-equal mapped error | Plan-equal rigid error | Mapped ≤1.5 mm | Wins on 145 support-matched near/wrong pairs |
| --- | ---: | ---: | ---: | ---: |
| Direct pose prior | 3.271 mm | 3.278 mm | 29.4% | 91 (62.8%) |
| Full atlas intensity | 3.131 mm | 3.135 mm | 31.8% | 94 (64.8%) |
| Atlas intensity zeroed | **3.113 mm** | **3.118 mm** | **32.3%** | 94 (64.8%) |
| Atlas intensity shuffled | 3.138 mm | 3.141 mm | 31.5% | 95 (65.5%) |
| Source locations shuffled | 3.151 mm | 3.154 mm | 31.1% | 94 (64.8%) |

The best fitted candidate in each beam averaged 1.255 mm and a ≤1.5-mm fitted candidate was available on 75.8% of eligible sections. The full score changed the prior's choice on 32/248, but reduced mean mapped error by only 0.139 mm and near selection by 2.4 percentage points. It was slightly *worse* than zeroing atlas intensity; the shuffled controls were similar. All four predeclared signal-gate conditions failed. The score therefore does **not** demonstrate anatomy-dependent fitting evidence, despite a mathematical path by which it could be backpropagated.

**Decision:** do not train the direct pose head with this energy or promote 132. The checkpoint's existing image and atlas features were trained for mapping, not for pointwise aligned-descriptor comparison; this is a specific negative result for using them as a fit metric. The next stage must train a geometry-blind source–atlas comparison signal on independent TRAIN sections using the **whole blind beam** and actual fixed fitted maps, then demand improvement over a matched support-only control before any fitting-to-pose feedback. Preserve the 132 weak-real coronal failure as a separate guard. All synthetic plans derive from one Allen template; this is not biological-animal validation, probability calibration, GUI qualification or a DeepSlice comparison.

The completed receipt at `I:/AnatomyTracker/runs/fixed_map_energy_133/completed.json` has SHA-256 `70f85944507c11ea3615c6409602c6c3c071e5fe40f315bf353a8fc268b87234`. Post-exit audit matched row, summary, config, source and protocol hashes; the raw output has 256 unique physical-section IDs, 248 complete 16-candidate rows, eight retained ineligible rows, 145 scored conditional pairs and finite selected errors.
