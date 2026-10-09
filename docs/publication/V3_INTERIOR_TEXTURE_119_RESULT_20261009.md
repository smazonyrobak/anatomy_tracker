# 119: acquired TRAIN tissue has more fine texture than the v3 generator

The [TRAIN-only audit](V3_INTERIOR_TEXTURE_119_PROTOCOL_20261009.md) completed on 512 donor-distinct acquired Allen sections and 256 fresh, independently drawn arbitrary-plane synthetic sections. Both were measured at 192². Its fixed threshold/closing/erosion tissue proxy yielded at least 1,024 interior pixels for 505/512 acquired images and 152/158 synthetic raw/exact-black images. On those 158 synthetic images, the proxy overlapped simulator-valid tissue with mean precision 0.891 and recall 0.994; no acquired tissue mask was available to verify it there.

| Mean interior metric | Acquired TRAIN | Synthetic raw/exact-black |
| --- | ---: | ---: |
| 10th–90th percentile intensity contrast | 0.558 | 0.524 |
| 1-pixel high-pass RMS / contrast | **0.0935** | **0.0751** |
| 3-pixel high-pass RMS / contrast | 0.2024 | 0.1866 |
| Proxy tissue area fraction | 0.217 | 0.263 |

The acquired fine-scale ratio is about 25% higher, and the gap persists descriptively within the common 10–40% proxy-area bins. This supports the concern that the synthetic source is too smooth at fine scale; it does **not** prove that texture is the cause of the millimetre-scale pose error, that a noise boost would fix anatomy, or that acquired slide defects have been replicated. Real images are mostly coronal and drawn from Allen experiments, whereas synthetic ones have arbitrary angles and a single Allen atlas source. The proxy and those composition differences limit causal interpretation. The separate 98 non-black/brush synthetic cases were not pooled into the primary acquired comparison.

Decision: keep arbitrary-angle sampling, but do not simply tune independent pixel noise to match one scalar. Use acquired TRAIN tissue patches and specimen provenance to design a small spatially plausible fine-texture/stain perturbation, with its own fresh synthetic and donor-separated real guardrail; prioritize the global pose-capture and candidate-selection failure in parallel. Acquired steep-oblique truth and final animal-level calibration remain missing. No checkpoint was trained, promoted or benchmarked here.

Post-exit audit found 768 rows, 512 distinct acquired donors, 256 distinct synthetic physical sections, and matching script/generator/real-manifest/prior-census/output hashes. SHA-256: rows `723c7ae949ea56f1f317ddf3910193e2925aa6279472d38b07fa7fc334cbc883`, summary `0cfc4248ce1614537530817eedb738b5d46c90fdf65da258d3eda92404b00378`, script `a7da8c46a758f4fa743595b00365f855298e7ec70dc77c1e14c758f001f11e05`. Frozen output: `I:/AnatomyTracker/runs/v3_interior_texture_119_train`.
