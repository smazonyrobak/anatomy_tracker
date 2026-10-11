# 239: consensus protects correct planes, but still misses the correction gate

The frozen no-training assay completed on 64 new TRAIN-inner sections (32 v4, 32 v7), with 64 distinct physical-section IDs and inputs. The independent post-exit audit matched the checkpoint, protocol, scripts, source and output hashes, draw count and all raw-row group means; the predeclared `stage_a_pass` is **false**. Truth picked a diagnostic near branch only after the blind top-16 beam existed. These are not blind model predictions or biological-animal validation.

| Metric | All | v4 | v7 |
|---|---:|---:|---:|
| Sections with `<2 mm` top-16 branch | 46/64 | 21/32 | 25/32 |
| Near initial mean tissue error (mm) | 1.306 | 1.186 | 1.407 |
| Near after intact-atlas consensus (mm) | 1.170 | 1.040 | 1.280 |
| Near after shuffled-intensity consensus (mm) | 1.285 | 1.148 | 1.401 |
| Known-true gauge, before / after intact fit (mm) | 0.182 / 0.173 | 0.126 / 0.119 | 0.238 / 0.226 |
| Near / true roles actually updated, intact | 11/46 / 3/64 | 4/21 / 1/32 | 7/25 / 2/32 |

The explicit no-shift hypothesis and geometric consensus solved the 237 true-pose drift: mean known-true tissue error did not rise. But the intact near gain is only **0.136 mm** and its matched null advantage is **0.115 mm**, below the frozen 0.25/0.15-mm requirements. Most near roles receive no update. Both appearance strata improve, but the conjunction still fails. This is another genuinely anatomy-sensitive but too-weak *conditional* correction, not a path to blind 3-D registration by simply training or tuning it longer. No thresholds were adjusted after seeing these rows; do not promote this fitter unchanged.

Together, 237–239 localize the bottleneck more precisely. The observed slice and atlas contain matchable internal texture, but independent normalized 5×5 patch matches are too ambiguous near the model's own imperfect planes. Outlier-resistant plane fitting prevents damage to a correct pose but cannot manufacture missing, spatially consistent anatomical correspondences. The next substantive change must learn stronger cross-modal, spatially coherent 2-D-to-3-D features on on-policy candidate neighborhoods, with source/atlas ablations that explicitly rule out support and position shortcuts. It must also improve or widen proposal capture; 238 found no `<2 mm` candidate among all 160 in 6/64 separate replayed sections. Reserve animal-level calibration and public benchmarking until those internal mechanisms and acquired-image guardrails pass.

Frozen output: `I:/AnatomyTracker/runs/local_3d_consensus_239_001`. SHA-256: `completed.json` `3e06896a27b1e664d2f26521563494cb6eda8f3543fb54e805d70c9d358113d1`; `summary.json` `6a8b1a590976e025766183d84649cb23734ea94f2789cada888c08c67ad9ddd2`; `rows.jsonl` `f0a79b5e89715c4a416720ed610760286cf36eba52dee0637b1044bc35fd82fe`.
