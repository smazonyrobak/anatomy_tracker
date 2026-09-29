# Signed pose evidence: one TRAIN activation measurement

Completed once, exit 0. The learned signed projection is small relative to the rest of the GRU input on this batch: **1.53–1.79% RMS** after aggregation over rows and feature pixels. The six raw cost-difference maps are not numerically vanishing. This supports an evidence-amplitude imbalance hypothesis, not a demonstrated cause of the failed pose-learning gate and not evidence that rescaling will improve accuracy.

## Frozen measurement

Whole `joint_v6_signed_pose_evidence_001/joint_model_step_02000.pt`; archived implementation at source commit `0e6dc0d8283c29ef64aabe4f2bdcdb080bd825b3`. Exact zero-based TRAIN schedule index **250**, the first large-perturbation batch: observation indices `[1464,101,19,753]`, modes `[raw,imperfect_brush,exact_black,raw]`, sections `[488,33,6,251]`. All four were eligible; they belong to three existing synthetic TRAIN subjects, not four independent biological animals.

One FP32 forward, B4/K1/R2/T3, AMP/TF32 off, both identity and horizontal-reflection branches with equal initial mass. No optimizer, backward, parameter/source modification, warm-up, repeated forward, sweep, development rows or new data. Only the original truth-near initialization and known PSF enter the model. Recorded truth reflection `[0,0,1,1]` is diagnostic metadata, never a model-selected branch or geometry input. No target ribbon fields were supplied.

Hooks saved all three raw six-channel maps and projected additions, all four GRU-input tensors, initial/state/update sequences, input channels, PSF, projection weights and row bindings. The fourth GRU call scores the final state and has no signed probe addition. Base evidence is defined by **GRU input minus the actual signed addition** and therefore includes image-pair plus coordinate evidence. RMS is over the entire feature canvas, not a tissue mask; aggregation uses root mean square, not a mean of row RMS values.

## Activation scale

| Pose-update iteration | Raster branch | Signed projection RMS | Base evidence RMS | Projection/base |
|---|---|---:|---:|---:|
| 0 | identity | 0.0052189635 | 0.3005671255 | 1.736372% |
| 0 | horizontal reflection | 0.0047796218 | 0.3119763611 | 1.532046% |
| 1 | identity | 0.0052285022 | 0.2949579437 | 1.772626% |
| 1 | horizontal reflection | 0.0049672247 | 0.3049591653 | 1.628816% |
| 2 | identity | 0.0052923607 | 0.2950527088 | 1.793700% |
| 2 | horizontal reflection | 0.0051851682 | 0.3024074455 | 1.714630% |

Raw cost-difference RMS pooled over iterations, rows, both branches and both signs is **0.082831515** for tangent-u, **0.067897895** for tangent-v and **0.076086029** for normal offset. The underlying six signed-channel, per-iteration/per-branch values and per-row values are in `result.json`; raw tensors have shape `(3,4,2,6,24,24)`. Per-row projection/base RMS spans **1.282432–2.621369%**. Projection-weight L2 norm is **0.728788131**.

Saved pose updates have RMS **0.001440841 rad**, **0.003815791 rad**, **4.071791 um** for the two normal-tangent coordinates and normal offset respectively, pooling all rows/branches/updates. Their maximum absolute values are **0.003214805 rad**, **0.010885770 rad**, **8.548968 um**. These are update amplitudes, not correction quality or a claim that either raster branch is correct. No replacement scaling or model intervention was applied.

## Receipts and limits

Output: `I:/AnatomyTracker/runs/joint_v6_signed_pose_activation_train_001/`. The measurement took 1.43324 instrumented forward seconds with peak CUDA allocation 778,859,520 bytes on RTX 2080 Ti; concurrent activity and no warm-up make this **not a throughput benchmark**.

The original selected-source archive omitted the transitive, disabled-uncertainty import `training/arbitrary_plane_joint_uncertainty.py`. Two initial setup attempts stopped before model construction or any forward: first that missing model import, then the raw-artifact loader's omitted transitive subject-slab import. The former was recovered from the **original signed source Git commit**, not current HEAD, into an isolated source supplement; the latter was avoided by direct authenticated metadata/NPZ field decoding. Historical archives were not changed. The actual supplement is saved as `archived_commit_uncertainty_source.py`; it matches the commit's contents modulo its terminal newline, with both byte hashes recorded. All other imported training modules resolve inside the authenticated signed source archive. This is one experimental forward, not three.

| Bound artifact | SHA-256 |
|---|---|
| Result JSON | `5762fdcda9d5eed305835b6e77c9f262710600ef97aa286fd1a1e167f091558b` |
| Raw hooks and pose tensors | `f34c58d5f7258d73f96505d644e7acc79cab1d081d8e7e5ef9fb66e5f0a5819e` |
| Measurement source | `8762c282bf344f662d58431733ba1b72bc4a343ba698fa75892e4c288630bba4` |
| Whole signed02000 checkpoint | `555c129b2cbe5832b37c7642d5ee15e31c1ba57199952901959751918cae6136` |
| Signed run completion | `7e814be7b2207e54544b67f6180a61808de17ea6bc530b41930092ced90828c1` |
| Exact original schedule | `e8aaa6eb1ff7e70097cbf1eeb1f6b9fc4d2ba4f01f26fc6d91b671c90c8dc0b9` |
| Observation identities | `1f3ea46ea294cf627b430a06cac7eea12cd32ffe762b6bcbf90b14319e0fb981` |
| Cohort completion | `ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2` |
| Saved imported source supplement | `e1989b684a72fd9af4a552066cc4e7d5c33cf2cb293513e628e52946962139e3` |
| Supplement's original Git blob | `733b5c1129a14ce1f8ddb300cacd93b781f5e1f6594169335071c61445678e09` |

`result.json` additionally binds the catalogue, each selected section metadata/NPZ, the full archived source manifest, exact initialization perturbations and persistent subject/animal/specimen/experiment/section/observation IDs. This conditional TRAIN diagnostic does not measure generalization, anatomical correctness, calibrated uncertainty or public-benchmark performance.
