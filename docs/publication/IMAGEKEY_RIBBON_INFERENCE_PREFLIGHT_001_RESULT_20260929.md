# Same-model image-key/ribbon inference — connectivity only

Session62858 exited0 from `d0eef25c78ed7f28294e384083ed3298c0a1d82a`.
The complete98,304-cell image-key posterior now feeds native ribbon refinement
through one functional entry point, preserving separate raster reflections.
This is **a numerical connection, not a working global-accuracy result**.

The sole input was original native TRAIN schedule observation592: synthetic
subject3, section5, `exact_black`. The frozen image, optional outline and known
synthetic PSF were passed unchanged; no truth pose, oracle reflection, support
mask, or deformation target entered inference. K=4 cells, R=2 representations,
T=3 updates and temperature.1 were fixed before execution. Chunk sizes1 and4
used the same whole native-no-probe step2,000 checkpoint. Its encoder/descriptor
tensors exactly match the original image-key4k gallery parent; no weights were
merged and no state tensor changed during inference.

## Numerical result and probability scope

- Coarse cell and initial component log probabilities were identical across
  chunkings. Selected cell/representation indices also agreed.
- Largest final centre/slab coordinate difference: **.0029296875um**; canonical
  residual difference.00020980835um; director difference3.5390e-8.
- Final conditional log-probability difference:9.5367432e-7. Reconstructed scores
  equal original full-catalogue component log mass plus the final refinement
  score **once**, with one normalization over all retained KxR components.
  Per-chunk normalized probabilities are not concatenated.
- Retained coarse mass: **.006889089558** (about.689%); omitted coarse mass:
  **.993111106345** (about99.311%). Their reconstructed sum1.000000195904 reflects
  FP32 normalization roundoff. The final retained conditional distribution sums
  to1, but this does not recover the omitted alternatives. The post-refinement
  tail is explicitly unknown, not assigned its old mass or declared zero.

Both distributions remain uncalibrated. No anatomical-error calculation, quality
threshold, held-subject evaluation, training or speed benchmark was performed.
In particular, this uses the same native control that **failed** its local
learning gate; the more successful real-data coarse branch was not spliced in.
The existing public model `forward()` is unchanged; the new functional entry
point is not yet the GUI runtime or a shipped replacement.

## Exact bindings

Output directory:
`I:/AnatomyTracker/runs/joint_v6_imagekey_ribbon_inference_preflight_001`.
All identities, source hashes, input pins, tolerances and raw predictions are
retained there.

| Artifact | SHA256 |
|---|---|
|completed.json|`75706cab713c3d794591b026d52b2c8ed837745ea12e3142a068f27b01b7e83e`|
|input.pt|`e692d865b50eed33c31c6f6c5ba9deef3cefcb938d2c46c57678e27193ed0415`|
|prediction_chunk_1.pt|`ac60e693d2efc496efb3d1a286c0f135d0807806e38eeebbcf8f012687b1d747`|
|prediction_chunk_4.pt|`3ea91a01b8b94ccebf8cfe725de5dae5c8f2e96d8f6ab89a6ecfe6e3c7400ca8`|
|arbitrary_plane_imagekey_ribbon_inference_v6.py|`a053202e29a8926154fa7f3bcfb2849cf1c021544aaba6e1850c1d073fed6eca`|
|preflight_joint_v6_imagekey_ribbon_inference.py|`32f20066ff6d3aa69e6669c54bb8972c98cbc8a95d7e2d51ec5e77ac0021b768`|
|Whole native step2,000 checkpoint|`966f236f98faa67cbc69c7b2236e12e84938d525496ed1a7644dbbc5e2745671`|
|Matching original4k gallery|`1568da8f1347641bba4ff93679ff39d5d6250693345db5d915322312bbe64666`|

## Remaining preparation

After the signed-evidence run completes, use its result to choose the local
learning mechanism, then freeze a predicted-candidate capture curriculum for one whole coarse model,
including the new physical TRAIN acquisition views after their corpus audit.
Do not infer that truth-near learning or this one-row connection closes the
finite-frame capture gap. Any changed encoder requires a matching rebuilt
gallery. Honest held-subject capture, treatment of omitted mass, uncertainty
calibration and actual GUI delivery remain separate unfinished work.
