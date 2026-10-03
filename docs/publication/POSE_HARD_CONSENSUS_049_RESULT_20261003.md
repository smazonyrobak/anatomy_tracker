# 049 hard-match consensus: small movement, wrong matches remain coherent

The query-selection check ruled out a simple empty-quadrant bug: the 041 visibility head placed an identity-equal mean 6.88 of the original eight quadrant-selected points on surviving synthetic tissue, versus 7.35/8 for unconstrained top-visibility selection. The 049 test used sixteen spatially distributed points, 82.6% of which were valid tissue on the frozen synthetic DEV panel.

With frozen scratch-lineage 019/041/046 weights, 049 re-ranked each query's 16 coarse atlas keys using 046, took a hard 3D point match, and fitted a full plane with all-point or fixed-seed robust consensus. The resulting equal-synthetic-identity *rigid tissue-coordinate* errors were:

| Fit | Top-one | Physically best of original eight (oracle diagnostic) |
| --- | ---: | ---: |
| Original 019 proposal | 2.499 mm | 1.001 mm |
| All hard matches | **2.437 mm** | 0.910 mm |
| Robust raw consensus | 2.440 mm | **0.909 mm** |
| Robust consensus with frozen 041 safety gate | 2.458 mm | 0.935 mm |

The largest best-eight gain is only **0.092 mm**, less than half the predeclared 0.2-mm threshold. On 64 weak-affine real DEV sections from six disjoint donors, prior top one was 0.584 mm; all-hard, robust-raw and gated top-one disagreements were 0.604, 0.610 and 0.582 mm. Worst donor regressions were 0.055, 0.065 and 0.016 mm respectively, all below the 0.2-mm limit, but the synthetic pose-capture gate still fails. The real references are inherited affines, not expert arbitrary-plane truth.

The failure is informative: 046 hard matches land within 1.5 mm of synthetic truth for 48.5% of valid queries on the prior top-one branch and 77.7% on the physically best prior branch, yet the fitted-consensus inlier fraction is 93–94%. Many **wrong atlas correspondences are mutually consistent**, so a high inlier count or small plane-fit residual cannot certify correct global anatomy. More robust fitting of these same matches, or another scalar candidate head over the same planes, is not the next priority. A new global image/atlas representation must distinguish distant anatomical lookalikes and improve near-truth pose capture before local warping and uncertainty can be trusted. Do not promote 049 to the GUI or infer calibrated region probabilities.

The read-only audit confirmed 177 frozen synthetic DEV rows from eight deformation identities and 64 weak-real rows from six donors, finite physical errors, the source hash and all config/row/summary SHA-256 receipts. The complete diagnostic is `I:/AnatomyTracker/runs/pose_hard_consensus_049_diagnostic`; row hash `a6108f886c0309acce74e2568c1001a59cce84a57f029f66c34af809a26b202f`. No rendered image was opened, no checkpoint was trained, and no public/final-test data were accessed.
