# 291 result: masking false atlas candidates does not repair correspondence

The frozen no-training control at `I:/AnatomyTracker/reports/masked_pair_cost_291_001` completed on all 64 cases of the 289 synthetic TRAIN-inner panel. Its eligible-site output contains 6,302 unique `(case, group, site)` rows from 49 cases; the other 15 cases had no site meeting the same displacement, support, and variance conditions as 290. The predeclared raw-score eligible counts, zero-offset and argmax hit rates, and mean argmax errors reproduced 290. The post-exit audit independently recounted rows and fragment hit rates and matched every source, protocol, configuration, and output SHA-256. The completed receipt SHA-256 is `ed3be7db66bc1e7f2c24263bd9e3aba7f0e3b137c1e46094787208e6d9560e49`; config is `5a539f26c3005b509e88b53382daec664e648326adfab319df50dce387da3686`; eligible-site table is `9e059a0d9e9839ed30ecb92dd539ec13864b003b8a1313fd20713dca931b1bad`.

| Prespecified 96-grid score | Intact: within 2 px, n=4,934 | Displaced fragment: within 2 px, n=1,368 |
| --- | ---: | ---: |
| Zero displacement | 75.25% | 2.70% |
| Raw 290 argmax | 35.53% | 13.89% |
| Exclude low-support/low-variance atlas candidates | 35.55% | 13.89% |
| Mask + 0.002 × squared offset prior | 50.99% | 19.30% |
| Mask + 0.010 × squared offset prior | 75.05% | 17.40% |

Only 0.22% of intact and 0.29% of fragment raw peaks were disallowed by the candidate mask. Thus the missing mask in 290 is real but is not the material cause of its poor argmax. The stronger central prior nearly reproduces the already-high **zero-motion** intact hit rate, not recovered tissue correspondence, and still selects the displaced-fragment offset within two pixels in fewer than one in five eligible sites. No fixed-score variant reaches the predeclared descriptive correspondence criterion from 290 (60% intact, 50% fragments, each ten points over zero). The proper next test is an explicitly supervised, learned multiscale local correspondence representation with a separate held-out synthetic panel and direct fragment-site readout; it should not merely enlarge the 289 decoder or tune a prior on this panel.

This is exact-pose synthetic TRAIN-inner evidence only. It neither demonstrates a learned-map improvement nor addresses blind pose, acquired-image generalization, steep-oblique real tissue, calibrated uncertainty, GUI qualification, or the sealed DeepSlice comparison. The 289 panel was reused solely for this no-training diagnostic, not as an independent model-selection set.
