# 305 acquired retention after fluorescence adaptation is mixed

The fixed no-training [TRAIN-only protocol](JOINT_SPATIAL_305_FLUORESCENCE_RETENTION_TRAIN_PROTOCOL_20261014.md) scored the batch-600 matched control and fluorescence-adapted checkpoints on exactly the same 58 Allen coronal and 40 Allen sagittal donor sections as the earlier 305 diagnostic. All 98 donor/animal/specimen/experiment/section/image identities and weak-reference axis fields matched the prior frozen cases. These upstream affines are approximate, not independently measured pose truth. No DEV/CAL/FINAL animal or public benchmark was accessed.

| Acquired TRAIN family | First-choice cardinal axis correct, control → adapt | Top-16 contains weak axis | Median unsigned normal gap | Median centre gap |
| --- | ---: | ---: | ---: | ---: |
| Coronal, 58 donors | 55 → 58 | 57 → 58 | 8.62° → 10.50° | 2.356 → 1.810 mm |
| Sagittal, 40 donors | **30 → 22** | 38 → 38 | 31.80° → 35.57° | 2.077 → 2.039 mm |

The separate fluorescence `AL1A` animal gain therefore does **not** qualify this batch-600 checkpoint for promotion: it trades a large near-coronal acquisition gain for a substantive first-choice sagittal-family regression. Top-16 sagittal axis coverage is retained, suggesting a ranking/mixture-weight effect more than complete proposal-family erasure, but this is an inference from the weak-reference readout, not demonstrated anatomy. Coronal orientation precision also worsens despite better cardinal-axis and centre selection.

The exact-pose local-map check on the same 64 frozen synthetic TRAIN-inner sections found only a +0.000346-mm adapted-versus-control mean change, so shared-feature drift did not materially damage that narrow mapping test. The previous 187 lineage showed that alternating older weak-real rehearsal can reduce such domain forgetting. The next *bounded* test should start from the adapted 305 checkpoint and compare matched continuations with and without coronal/sagittal rehearsal while retaining independently sampled full-sphere synthetic and Lupori TRAIN images. If rehearsal does not restore sagittal choice without harming `AL1A` or synthetic retention, do not scale it. None of these near-cardinal weak references validates physically cut steep-oblique brains, joint fit-to-pose feedback, calibration, or GUI use.

Frozen output: `I:/AnatomyTracker/runs/joint_spatial_305_fluorescence_retention_train_001`; completion SHA-256 `008ed44bcd232bcd479c49bf0a4caa76d8851b5488b5abb77d70e641c13317dc`. The script and protocol were committed in `6a3a0b5` before this run; the shared-index commit also contained an unrelated exact-map result document, but no scientific output was modified.
