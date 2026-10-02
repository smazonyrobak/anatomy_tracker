# Atlas-MIND ranking probe 016: frozen failure on actual candidates

The model-free 64-case pilot's high MIND rank on a constructed near-truth candidate bank did **not** transfer to the current one-shot model's actual 32 predicted pose/reflection branches. With the 015 batch-8,000 checkpoint frozen, each of the same 185 eligible synthetic DEV sections had all 32 planes rendered from the pinned Allen atlas at 96 pixels with its finite-thickness PSF. No weights or poses were updated. The descriptor used the pre-existing physical MIND implementation; the two masked variants penalized predicted tissue outside candidate atlas support. The true-tissue mask is a privileged, non-deployable upper bound.

| Selector on identical 32-branch pool | Plan-equal rigid visible-tissue error | <5%-tissue error |
| --- | ---: | ---: |
| Image-only direct prior | **2.811 mm** | **5.098 mm** |
| Whole-image MIND | 3.378 mm | 5.803 mm |
| Predicted-mask support-aware MIND | 4.058 mm | 5.989 mm |
| True-mask support-aware MIND, diagnostic only | 3.809 mm | 6.229 mm |
| Physical-error oracle, unavailable in use | 0.970 mm | 1.717 mm |

The deployable predicted-mask score missed the preregistered 0.2 mm improvement gate by a wide margin and worsened severe partial cases. It improved 59/185 sections, worsened 125, and tied one relative to the direct prior. Even the true-mask diagnostic failed, so this is not merely a tissue-mask threshold problem. Within-case score versus negative physical-error Spearman correlation averaged only 0.159 predicted-mask, 0.203 true-mask and 0.209 whole-image. These are weak associations, not likelihoods or calibrated uncertainty. The 32-branch oracle remains much better than selection, but even it is not consistently submillimetre on the most partial sections.

**Decision:** reject this MIND scalar as a global pose selector and do not wrap it in a larger search or use it as a fit-feedback likelihood. The old constructed-bank pilot demonstrated that precise nearby planes can contain discriminative image information; it did not establish a useful energy landscape among millimetre-wrong model proposals. More score-only loss or privileged masking would not be a principled next investment. The next training change should enlarge the image representation and proposal capacity with substantially more diverse arbitrary-plane exposure, while monitoring direct physical DEV error before committing to a million-presentation run. No GUI promotion, final-test or public DeepSlice benchmark follows from this probe.

Frozen diagnostic: `I:/AnatomyTracker/runs/one_shot_mind_candidates_016`. Its read-only verifier matched the source, MIND implementation, checkpoint, DEV panel, raw-row and summary hashes; it recomputed all 5,920 candidate physical-error/score choices and ten reported plan/support means from 185 raw sections. This is synthetic development evidence from eight deformation plans, not eight biological animals.
