# Fixed TRAIN RMS scaling of signed pose evidence

Matched control **completed from `2a86507`, session90795 EXIT0**, after its
compatibility preflight passed. Output `I:/AnatomyTracker/runs/joint_v6_normalized_signed_pose_evidence_001`
is frozen and independently audited; the scientific normal-learning gate fails.
This changes one input scaling, not
the architecture, feature encoder, optimizer, loss, rendering or topology limits.
All work stays on I:. The separate full-coverage coarse run also completed.

## Evidence and hypothesis

The [unscaled signed control](NATIVE_SIGNED_POSE_EVIDENCE_001_RESULT_20260929.md)
failed: oracle centre reduction12.81%, normal reduction.594%, versus20% required
for each. The [TRAIN oracle diagnostic](TRAIN_POSE_COST_DIRECTION_001_RESULT_20260929.md)
nevertheless found useful full-map cost directions128/144 times. On one actual
scheduled coupled-perturbation TRAIN batch, the learned signed addition was only
1.532–1.794% of the other GRU evidence RMS.
[Measurement and limitations](SIGNED_POSE_TRAIN_ACTIVATION_001_RESULT_20260929.md).
These observations motivate preconditioning; they do not prove that amplitude
caused the failed learning or that normalization will fix it.

## Single fixed intervention

Divide the existing six signed maps before their existing zero-initialized
projection by `[ru,ru,rv,rv,rd,rd]`, where

```
ru = 0.08283151464815028
rv = 0.0678978954706466
rd = 0.07608602924318171
```

These are the complete-canvas raw-map RMS values pooled across both signs,
three iterations, four TRAIN rows and both reflection branches in the completed
measurement. Its result SHA256 is
`5762fdcda9d5eed305835b6e77c9f262710600ef97aa286fd1a1e167f091558b`.
Rows1464/101/19/753 are exactly schedule index250, fixed before measurement.
No development example, quality score or scale sweep determines these values.
The measurement uses the trained unscaled model's states; it is a small TRAIN
scaling estimate, not a population statistic or a calibrated uncertainty scale.

The positive shared-sign divisor preserves each directional comparison. Do not
demean, normalize signs separately, adapt scales per sample or supply oracle
support. Keep the same constants at inference. The optional constructor value
is saved in model configuration; `None` preserves the old arithmetic and state
dictionary. No new parameters, persistent buffers or random draws are introduced.

## Matched learning and decision

Completed: runner90795 exited0 and the independent audit passed integrity but
failed the unchanged normal-learning gate. See the
[result and continuation decision](NORMALIZED_SIGNED_POSE_EVIDENCE_001_RESULT_20260930.md).

Initialize the **whole original image-key4k model**, not the trained signed2k
weights and not any separately trained coarse encoder. Fresh native heads,
zero coordinate/ribbon/signed heads, seed2026092913, exact original schedule,
2,000 updates, B4/T3/one pose-only prefix, frozen retrieval, AdamW2e-4,
weight decay1e-4 and gradient clip1 remain unchanged. Preserve all original
physical losses, known PSFs, reflection handling and perturbation curricula.
The completed unscaled signed control is the primary matched comparator;
the no-probe result remains a secondary descriptive reference.

Keep the exact original signed gate:20% selected/oracle centre and slab
improvement versus the geometric initializer,20% oracle-normal improvement,
nonregression for each populated eligible mode/held-out synthetic subject,
finite outputs, original topology/orientation and unchanged retrieval tensors.
Report differences from the unscaled comparator, not only initializer-relative
improvement. No threshold waiver, scalar sweep or extra updates after inspection.

One fixed TRAIN-batch compatibility preflight compares unscaled versus scaled
fresh models: identical state keys/parameters/RNG and zero-projection outputs,
finite scaled forward/backward, no optimizer update. Archive all operative
sources including the historically omitted uncertainty-module import. Bind that
receipt and commit the runner before launch. Save original schedules, sources,
checkpoints and per-row raw predictions; audit only after confirmed exit.

A passing local gate would still not establish honest global retrieval,
biological generalization, calibration, public-benchmark superiority or GUI
readiness. Any future successful coarse model needs native training within its
own whole-model lineage; this control's refiner cannot be spliced into it.

## Completed compatibility preflight

Session17128 exited0 from `e6c97e6`. Fresh unscaled/scaled models had identical
parameters/state keys/CPU RNG and exact zero-head outputs; both used22 forward
atlas encodes. One scaled backward was finite, signed-projection gradient
norm.1785823554, with no optimizer or parameter changes. Source/model proof is
the new-code None-versus-RMS zero-head comparison plus the reviewed default-path
diff, not a separate archived-legacy numerical replay. Receipt:
`I:/AnatomyTracker/runs/joint_v6_normalized_signed_pose_evidence_preflight_001/preflight.json`,
SHA256 `1d87df5a33bd2da6df570a84108b9e9a5b0fec888ef10f4c52578b8f7f4d68c4`.
The matched runner pins this receipt, the failed comparator audit and original
TRAIN activation scales, and requires matching shared operative source hashes.
