# Matched short v10-framing pose continuation result (14 October 2026)

**Decision: fail; do not promote the 0.72 FOV recipe or extend this branch.** The predeclared primary comparison required a ≥0.20-mm first-choice error reduction on fresh v10 sections. After 400 four-image batches per arm, control was **2.850 mm** and the v10-slot intervention was **2.888 mm** (0.038 mm worse). Both arms began at the same **2.823 mm** parent value on that panel. The new panel had 6/64 sparse (<15% valid tissue) and 11/64 ≥45° oblique sections, so the prespecified coverage condition was satisfied.

The two arms used the same parent and AdamW moments, old/acquired training schedule, loss formula and frozen v9/v10 evaluation panels. They trained on **800 distinct synthetic physical sections per arm**, with disjoint synthetic seeds and no deliberate same-plane background/artifact copies. The control used two v9 sections per batch; the intervention used one v9 and one v10. Each arm also saw 400 Lupori TRAIN and 400 older TRAIN presentations. The parent checkpoint is descended solely from project random initialization, not external weights. No acquired DEV/CAL/FINAL, public benchmark, calibration, or GUI deployment was touched. Frozen results: `I:/AnatomyTracker/runs/fov_balance_v10_short_pose_001`.

| Frozen readout, batch 400 | Control | V10-slot arm |
|---|---:|---:|
| Fresh v10 first-choice rigid-site gap | 2.850 mm | 2.888 mm |
| Fresh v10 top-16 capture within 1.5 mm | 47/64 | 49/64 |
| Fresh v9 first-choice rigid-site gap | 2.698 mm | 2.473 mm |
| AL1A weak Visualign five-site gap | 1.656 mm | 1.630 mm |
| Fully held-out TRAIN-set sagittal donor axis | 19/20 | 19/20 |
| Fully held-out TRAIN-set coronal donor axis | 20/20 | 20/20 |

Six/eight v10 synthetic bases and 44/64 v10 sections improved, but the result is dominated by an important genuine tail failure: one 7.1%-visible, box-offset, TRAIN-framed section changed from **4.39 to 10.89 mm** first-choice error. Its best top-16 candidate was still ~3.06 mm away, so a selector alone cannot recover that case. Removing it or switching from the preregistered mean to a median would silently discard precisely the sparse-slice behavior the model must handle. The other ten gate clauses passed; the primary mean-error clause did not. The 400-batch synthetic training loss also showed no clear downward trend for the v10-slot arm (100-batch means ~7.55, 8.00, 7.59, 7.96), so this is not evidence for simply buying more v10 exposure.

A subsequent exploratory read-only decomposition of the frozen parent and panels (not part of the preregistered gate) found that the model already emitted **80 pose modes × two reflections**. For the v10 panel, 48/64 had a ≤1.5-mm candidate within the top 16, but only **51/64 had one anywhere among all 160 branches**: three misses were ranking, while thirteen were missing proposal geometry. On the independent v9 panel the corresponding counts were 58/64 and 58/64. The v9/v10 panels contain different sections, so the between-panel difference is descriptive, not a causal FOV effect. At least on v10, simply reranking the existing outputs cannot solve most misses.

An independent post-exit audit passed: committed runner/protocol/generator and parent bindings, all parent/top-level/arm output hashes, 128 panel accepted identities (eight per base and recipe), 1,600 accepted distinct training synthetic identities, disjoint panel/training seeds, 400 batch and three readout records per arm, per-case metric reconstruction, gate arithmetic, and a model-state comparison showing only the allowed encoder/lateral/pose/anchor tensors changed. The first parent-hash assertion in the runner contained a transcription typo and stopped before any output or training; correction was committed as `b90dedf` before the successful run. The protocol hash itself was correct.

This points away from FOV-fraction matching as the next lever. The relevant bottleneck remains whether blind proposals contain a correct arbitrary plane, especially with small/ambiguous tissue fragments, and whether the model can flag insufficient visual information. Next work should target proposal coverage and calibrated abstention/uncertainty using independently sampled cuts, with acquired-domain guardrails; do not launch full-scale training or DeepSlice comparison on this checkpoint.
