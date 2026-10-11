# 193: real-field adaptation repairs the inherited map, but not beyond zero warp

The [bounded protocol](LUPORI_LOCAL_MAP_ADAPTATION_193_PROTOCOL_20261011.md) completed 600 batches on the three allowed Lupori TRAIN-gradient animals. Each batch presented one independently drawn full-sphere v4 synthetic section to retain arbitrary-angle mapping. Only the 187 step-800 model's `warp`, `warp_condition`, `pair` and `atlas_encoder` modules changed; image encoder, probabilistic pose heads and uncertainty remained frozen. The 52 valid sections of `AL1A` were excluded from gradients. No DEV/CAL/FINAL animal or public benchmark was read.

On that entire inner animal, with the approximate human plane supplied, the local-vector RMS relative to the affine-free VisuAlign field was:

| field | mean / median / p90 (mm) |
|---|---:|
| no local deformation | **0.1056 / 0.1020 / 0.1553** |
| inherited parent map | 0.1535 / 0.1269 / 0.2258 |
| adapted step-600 map | 0.1030 / **0.0987** / **0.1480** |

The adapted map reduced the inherited-parent mean error by 33%, and no section was >0.1 mm worse than zero. But it beat the strong zero-field baseline by only **2.4%**, below the predeclared 10% requirement. The real-map gate therefore **failed**. Both real and synthetic within-training errors declined over the run (first→last 100 batches: 0.1056→0.0979 mm real, 0.475→0.440 mm synthetic), but no frozen synthetic retention panel was read because the real guard did not pass. These losses are not a substitute for held-animal accuracy.

The first guard invocation stopped before reading images because the PyTorch checkpoint preserved a tuple where JSON preserved a list; a committed serialization-normalization fix reran the same frozen checkpoint and produced the sole completed guard record. The independent read-only verifier checked all 600 gradient/draw records, 188 distinct allowed real sections, 600 distinct sampled synthetic physical IDs, source image/field/mask hashes, checkpoint/source/protocol hashes, all 52 guard source hashes, and summary arithmetic.

**Decision:** the adapted field is a safer research starting point than the inherited field, but it has not learned reliable individual-specific local deformation beyond applying no warp. Do not promote it, derive fitting difficulty from its magnitude, claim calibrated electrode-site regions, or trigger a full benchmark. The 188 pose model remains useful for candidate capture, but the requested joint pose/deformation feedback is still scientifically unverified. Better real spatial supervision and more independent animals—including physically steep-oblique cuts—are the next binding needs; additional batches on these same three weakly labelled animals are not justified by the guard result.

Frozen training completion SHA-256: `0f934797603d6dc5a991bdfaba7d8dcc5c15906bd9c5f222f34d8d3a017d15fc`; frozen AL1A guard completion SHA-256: `6ecc86674d640cab10b9b044c356897c6c9f9b9c09aa99197d00bf4c5840ab4b`.
