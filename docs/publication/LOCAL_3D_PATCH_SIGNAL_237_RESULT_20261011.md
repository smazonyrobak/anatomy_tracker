# 237 local 3-D patch signal: failed mechanism gate

The frozen 64-section TRAIN-inner assay completed without training updates. The independent verifier passed the protocol, code, checkpoint, input, raw-row and summary hashes, confirmed the same physical-section IDs and image hashes as 235, recomputed the reported metrics and the gate, and found **`stage_a_pass: false`**. These are conditional diagnostic branches selected using truth after the blind top-16 beam was generated, not blind predictions or independent-animal validation.

| Measure | All | v4 | v7 |
|---|---:|---:|---:|
| Sections | 64 | 32 | 32 |
| Top-16 with a `<2 mm` branch | 51 | 26 | 25 |
| Comparable near/wrong pairs | 39 | 20 | 19 |
| Near branch initial mean tissue error (mm) | 1.360 | 1.296 | 1.426 |
| Near branch after intact-atlas full-pose fit (mm) | 1.211 | 1.123 | 1.302 |
| Near branch after intensity-shuffled null fit (mm) | 1.324 | 1.243 | 1.409 |
| True-pose drift after intact-atlas fit (mm) | 0.421 | 0.368 | 0.475 |
| Near/wrong score wins, intact | 25/39 | 12/20 | 13/19 |
| Near/wrong score wins, null | 21/39 | 10/20 | 11/19 |

The intact fit gains **0.149 mm** from a truth-selected near branch and beats the matched null by **0.113 mm**. Both appearance modes improve, but neither gain meets the predeclared 0.25/0.15-mm thresholds. The 0.421-mm drift from known-true planes exceeds the 0.20-mm ceiling. These failures independently reject the mechanism for promotion or training unchanged. Thirteen of 64 sections have no `<2 mm` top-16 branch; no local updater could fix their missing-candidate problem without a larger capture range or improved global retrieval.

There is nevertheless real image-anatomy information in the patch matcher. Averaging the per-section fractions of matched, supported source sites, intact-atlas matches at near branches land within 0.5/1.5 mm of truth in 0.141/0.409 of sites, versus 0.026/0.229 for shuffled intensity. At the true pose those fractions are 0.471/0.684 intact versus 0.054/0.310 null. Thus the failure is not evidence that atlas texture carries no signal. The spatially inconsistent matches and robust plane fit cannot currently turn that signal into a reliable pose correction; the true-pose drift makes direct use unsafe. The score orders a near branch ahead of a support-comparable wrong branch in only 25/39 pairs, so this is also not a useful blind reranker as-is.

Next work should jointly address (1) coherent, no-drift anatomical correspondence/pose correction and (2) missing near candidates. A candidate-conditioned recurrent updater is justified only after a small matched test demonstrates stable correction on actual blind beams and anatomical advantage over the support-preserving null. The acquired-image and physically oblique animal evidence remains insufficient to claim arbitrary-angle real-world reliability. No DEV/CAL/FINAL animals, DeepSlice benchmark, GUI deployment, or probability calibration were used in 237.

Frozen artifacts: `I:/AnatomyTracker/runs/local_3d_patch_signal_237_001`. SHA-256: `completed.json` `ad58e5ad1265173268bbb5b25b9c45c26db3656dc7a676225780810b8dc7e301`; `summary.json` `cdabd3ab8935a64c890e75e2045ead2bbde411b5e3d3e0550fb4f64e51760f5a`; `rows.jsonl` `d7987a925863e6766bfa07997a94538a4c6b3ac70bf6a9712d4679396b11d20e`. The reproducible audit is `training/verify_local_3d_patch_signal_237.py`; the pre-result protocol is `LOCAL_3D_PATCH_SIGNAL_237_PROTOCOL_20261011.md`.
