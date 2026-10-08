# 101: better local matching barely changes the selected plane

The pre-score protocol and script are `MATCHER_SWAP_101_PROTOCOL_20261008.md` and `training/diagnose_matcher_swap_101.py` at commit `b43e15f`. The 101 runner exited 0. Output is `I:/AnatomyTracker/runs/matcher_swap_101_development_diagnostic`; its authenticated `summary.json` SHA-256 is `10aa618dc1c6cdce70670267f710fd491b82eea35d08ea9a2aeb104b525bc764`. Independent checks matched all four output hashes, source/protocol hashes, 1,792 candidate rows and 128 section-arm rows, unique 64 section IDs, all selected/best slots recomputed from raw rows, and physical-error means recomputed from those rows. The first section's zero-context arm had **exact zero** difference from the frozen 094 fitter for match logits/support, fitted state, spatial evidence, quality/coverage, mapped CCF and selection score.

| Same 64 synthetic DEV sections, 14 fixed candidate branches | Original 094 matcher / 099 step 0 | Trained 099 matcher / step 4000 |
| --- | ---: | ---: |
| Score-selected mapped visible-tissue error | 2.908 mm | 2.847 mm |
| Physically best fitted candidate error | 0.965 mm | 0.968 mm |
| Selection regret | 1.943 mm | 1.879 mm |
| Score-selected normal error | 31.50° | 31.40° |
| Score-selected centre error | 2.197 mm | 2.205 mm |
| Fit energy | 0.327 | 0.278 |
| Fitted quality | 0.745 | 0.783 |

The selected branch changed in **only 1/64** sections. Five of eight synthetic-plan means improved, three worsened; one improvement was only 0.002 mm. Coronal-like, horizontal-like and sagittal-like family changes were −0.157, +0.010 and −0.028 mm respectively. The score gap between selected and physically best branches averaged roughly 0.75, almost entirely tracking the prior-score gap; the mean fit-energy difference between them was only about −0.005 in the initial arm and −0.004 in the updated arm. Thus the lower fit energy and higher fitted quality are not reliable evidence of a better selected *physical* plane. The matcher improvement documented in 099 did not meaningfully propagate through the frozen 094 fitting/selection head.

This does **not** establish that matching feedback is useless, that arbitrary cutting planes are impossible, or that the new descriptor cannot help when the fit/selection head is jointly trained. It does reject an immediate 099 matcher swap as a solution and rejects scaling its descriptor-only objective by itself. Next: diagnose and train the explicit link from atlas-match evidence and bounded deformation difficulty to candidate posterior/pose correction on fresh TRAIN draws, with physical candidate supervision and a newly frozen identity-disjoint confirmation panel. Preserve the unchanged candidate beam and no-true-mask inference, compare score-selected physical error and normal error, and stop if the fitted score still rewards wrong but easily warped atlas planes. Real oblique animal data, calibrated uncertainty, GUI promotion and public DeepSlice benchmarking remain separate later gates.
