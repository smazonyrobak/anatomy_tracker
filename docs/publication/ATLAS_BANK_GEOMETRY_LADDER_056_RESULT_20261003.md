# 056 result: the true atlas key loses mostly before position quantization

The scratch-trained 025 query and atlas descriptors were held fixed. On 185 synthetic DEV sections/eight held-out deformation identities, the same 32 seeded tissue-valid points per section were compared against increasingly realistic atlas patches. Each image query was placed in the antipodal bank hemisphere by the same horizontal parity operation used in retrieval; all reported cosine similarities are identity-equal means. No images were visually inspected.

| Atlas patch paired with the same observed query | Mean cosine |
| --- | ---: |
| A. Exact oracle-warped per-pixel surface | 0.898 |
| B. Rigid patch at true CCF point, exact observed basis and section PSF | 0.797 |
| C. Same true CCF point, nearest bank frame + fixed 67 µm/pixel + bank PSF | 0.548 |
| D. Actual bank key at nearest stored grid point and frame | 0.502 |
| Highest-scoring key anywhere in the bank | 0.876 |

Allowing the best of four nearby bank frames raises C to 0.657 and D to 0.625, still far below the average global top score. The nearest bank position is 248 µm from the true point on average; its nearest frame differs by 13.7° on average, while the observed image's median physical pixel scale is close to the bank's 67 µm. For the best of those four geometrically nearby frames, the actual correct-location bank key appears in the global top 16 for only **2.98%** of queries, and its pooled median rank is 7,384 among 2,061,824 keys.

The A→B 0.100 cosine reduction reflects removal of local warp from the atlas patch while keeping the anatomical centre correct. The larger B→C 0.249 reduction combines **orientation, scale, shear and through-plane PSF discretization**; this test does not yet identify which of those changes dominates. C→D adds a smaller 0.046 reduction from snapping to the stored position grid. The 030 result of 15.42% correct-location top-16 recall is higher than the 2.98% here because it accepts *any* of 512 frames at the correct position; a mismatched frame can occasionally score well. That does not establish a correct plane orientation.

These results explain why the rigid-local 053 control (75% top-one among 288 known-plane candidates) cannot be extrapolated to a whole-brain atlas bank. The true match is often too weak after bank geometry changes, before the millions of unrelated lookalikes compete. Merely training another scalar score or adding a few more nearby grid positions would not address the dominant B→C drop. A matched numerical factorial of frame/scale/PSF changes is the next targeted decision before paying for a denser bank or more training.

The evaluator exited normally. Read-back verified config, row and summary hashes; all 185 rows match the frozen eligible DEV panel, with exactly 32 queries each and preserved synthetic animal/specimen/experiment/section IDs. The bank and model are unchanged and were trained from random initialization; no real final-test animal, public benchmark, calibrated probability or GUI replacement was used. GT-valid query selection and known truth geometry make this an oracle diagnostic, not an inference result.

Frozen output: `I:/AnatomyTracker/runs/atlas_bank_geometry_ladder_056_development_eval`.
