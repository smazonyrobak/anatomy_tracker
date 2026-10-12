# 275: more ordinary synthetic exposure improves capture, not selection or real transfer

The frozen 275 run completed 8,000 additional optimizer batches of four independently drawn v9 synthetic planes: 32,000 accepted presentations from 35,197 attempts. It continued this project's own randomly initialized 274 **control** checkpoint and AdamW state, without acquired-image labels, external weights/features/pseudolabels, DEV/CAL/FINAL examples, or public benchmark. A new 64-case TRAIN-inner panel was fixed before updates, with eight independently drawn eligible planes from each of eight held-gradient *synthetic deformation bases*. Two older panels and 50 odd-hash acquired TRAIN donor median sections were held fixed. These are not 64 or 50 blinded biological test animals.

The independent `training.verify_pose_capture_275` audit passed all frozen file/parent/source/protocol hashes, 64 panel records, 35,197 draw records, 8,000 ordered training records, 20 row-level readout blocks, exact checkpoint selection/gate arithmetic, and 24 freshly recomputed image-to-pose case readouts. Selected batch 8,000 checkpoint SHA-256: `b4689e1d0dcbf1e8d65a63a400962bf27f44b2e571c8361e29e0e51412b2b97f`.

| Added batches | New v9 top-16 ≤1.5 mm | New mean first-choice site error | New mean normal-axis error | Frozen 270 top-16 | Frozen 268 top-16 | Acquired coronal family | Acquired sagittal family |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 50/64 | 2.267 mm | 34.1° | 48/64 | 53/64 | 28/28 | 13/22 |
| 2,000 | 54/64 | 2.577 mm | 32.4° | 48/64 | 55/64 | 28/28 | 6/22 |
| 4,000 | 55/64 | 2.609 mm | 33.2° | 49/64 | 56/64 | 27/28 | 3/22 |
| 8,000 | 60/64 | 2.329 mm | 31.2° | 56/64 | 61/64 | 25/28 | 9/22 |

The predeclared selected-checkpoint gate **fails three conditions**: new-panel mean first-choice error is 2.329 rather than ≤2.300 mm; acquired coronal family is 25 rather than ≥26/28; acquired sagittal family is 9 rather than ≥16/22. It passes new and old top-16 capture. Do not round the 2.329-mm miss into a pass. Further unchanged v9-only scaling is blocked: more exposure clearly creates nearby candidate planes but does not reliably rank them first, and its transfer to the physically sagittal Allen images deteriorates. The acquired family reference is an automated weak Allen affine, not expert ground truth, but the fall from 13/22 to 9/22 on the same donors is a real diagnostic warning.

At selected batch, 12 newly drawn cases show <15% visible tissue: 11/12 have a near plane in the top 16, but mean first-choice error remains 2.669 mm. The 45 cases with 15–40% tissue have 42/45 top-16 capture and 2.404-mm first-choice error; the seven with ≥40% tissue have 7/7 and 1.264 mm. Of nine planes more than 45° from every cardinal normal, eight have a near top-16 candidate yet mean first-choice error is **3.953 mm**. This is synthetic evidence that heavy obliques are represented in the proposal set, not proof of usable arbitrary-oblique localization; the ranking gap remains severe. A tiny or damaged fragment can be physically ambiguous, requiring calibrated uncertainty or abstention rather than a fabricated certain coordinate.

Next, the already committed zero-training 276 diagnostic asks whether summing the model's mode mass by antipodal normal family changes the wrong first-choice decision. If its fixed gate fails, the result points to a richer anatomy/pose-selection mechanism and/or genuine coarse real-image supervision, not another v9-only long run. Correct-pose local mapping, joint fit-to-pose feedback, optional acquisition-constraint conditioning, independent expert animal-level and steep-oblique validation, calibrated electrode-site/region probabilities, and GUI packaging remain undone. DeepSlice comparison remains premature.

Frozen output: `I:/AnatomyTracker/runs/pose_capture_275_v9_continuation_001`. Protocol: `docs/publication/POSE_CAPTURE_275_PROTOCOL_20261012.md`.
