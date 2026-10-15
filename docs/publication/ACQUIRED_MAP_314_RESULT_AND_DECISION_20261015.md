# 314: current-lineage acquired-map guard and development decision

The [frozen protocol](ACQUIRED_MAP_314_PROTOCOL_20261015.md) was committed and pushed as `c98b035` before this readout. The frozen [script](../../training/assay_acquired_map_314_AL1A.py) evaluated the 309 parent and both 312 arms with no optimizer update, on all 52 valid acquired sections from Lupori animal `AL1A`. This is a TRAIN-inner animal absent from 309/312 map gradients, **not** official held-out validation. Each mapper was given the human approximate QuickNII plane, so this isolates conditional local-field quality rather than blind plane recovery. The reference is the human VisuAlign local correction after the model's unweighted affine-free gauge; the zero field is scored on the same 96-pixel tissue sites and current 192-pixel input grid.

| Frozen predictor | Equal-section mean tissue vector RMS (mm) | Median (mm) | 90th percentile (mm) | Predeclared gate |
| --- | ---: | ---: | ---: | --- |
| Zero local displacement | 0.105496 | 0.102032 | 0.155107 | baseline |
| 309 parent | 0.110018 | 0.106791 | 0.153188 | fail |
| 312 detached control | 0.110061 | 0.104556 | 0.155918 | fail |
| 312 joint-feedback | 0.109672 | 0.104174 | 0.155293 | fail |

The best learned field, 312 joint-feedback, is still 0.004175 mm (3.96%) worse than zero on the mean. It is only 0.000347 mm better than 309, far below the predeclared requirement of at least 10% improvement over **both** zero and 309. No model had a section >0.1 mm worse than zero, but that tail condition alone cannot pass the gate. This result does not prove that local deformation is unlearnable; it says the **current** field supplies no demonstrated acquired-image advantage and its deformation/fit cost must not be treated as reliable pose feedback. The earlier 192/193 acquired-field results pointed in the same direction, though those used an older lineage and a different input sampling grid.

Independent read-only audit passed: 52 unique valid `AL1A` section IDs exactly match the source metadata; all 208 thumbnail, X/Y displacement CSV, and mask hashes match the 48,652,184,187-byte archive; all three checkpoint hashes, script/protocol/split hashes, output hashes, per-row arithmetic, medians, 90th percentiles, and gate booleans were recomputed. Frozen output: `I:/AnatomyTracker/reports/acquired_map_314_AL1A/{config.json,rows.jsonl,summary.json,completed.json}`. Script SHA-256 `0f5f48f1166249f74a5f8f26c45a794af8e537ffd57d4cbbc9cabaab7467ce80`; rows SHA-256 `02044d9eb124ea85a8f10a364a6910fc9f8325c500076d7ff1ef4a2fdb082f11`; summary SHA-256 `41276119b4beeef7ef0e499bef5a8736c252366ec18f8a4201fce52c48586831`.

## Development decision

Stop scaling the unchanged 309/312 loss or training longer on the same synthetic map supervision. The matched 312 pose-feedback pilot had a joint-minus-detached first-choice mean gain of only 0.0286 mm with a paired interval crossing zero; this 314 acquired-map gate fails even with a supplied plane. Both are insufficient evidence for a more expensive feedback architecture. The 309 synthetic first-choice rigid site gap remains 2.469 mm, although its exact-pose synthetic local residual is 0.109 mm, so longer synthetic exposure alone is not a demonstrated route to a shippable model.

The next work should be a **data-and-label decision**, not another optimizer variant: inspect representative acquired source images and their human plane/field overlays for label quality and image-domain mismatch, with animal IDs preserved; identify a credible source of physical high-obliquity plane truth; only then change the simulator or supervision where those observations show a specific gap. A small animal-separated acquired-map/pose readout should follow any such change before scaling presentations. Reserve official DEV/CAL/FINAL animals and DeepSlice benchmarking until an internally promising model exists. Do not claim calibrated region probabilities or GUI readiness from this assay.
