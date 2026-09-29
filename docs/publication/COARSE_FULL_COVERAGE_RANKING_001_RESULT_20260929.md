# Full-coverage coarse ranking: uniform arm completed

The completed independent CPU audit confirms **integrity passed and all 12 fixed engineering gates passed** for arm A. Under the preregistered sequential policy, arm B was deliberately **not run**: there is no hard-versus-uniform negative-mining result.

This is useful coarse candidate retrieval, not a qualified alignment/deformation model. Arbitrary-plane synthetic selected-angle error remains large, real selected-frame error remains millimetric, and an image-free TRAIN orientation prior beats the model's real selected-normal error. No public benchmark or calibration was performed.

## What was trained

Whole own-lineage experimental C10000 checkpoint plus its AdamW/RNG; no encoder/refiner splicing or external weights/features/pseudolabels. Exactly **12,576 additional applied updates**, ending at step **22,576**, FP32/TF32-off, LR0.00025, with eight real and eight synthetic queries per update. Both image stems, shared encoder and image descriptor trained; the runner reports other whole-model tensors preserved exactly. This does not turn the concurrently trained native model into this checkpoint's refiner.

The stage combined expanded real TRAIN data (1,280 images, the same 58 donors), fresh appearances, remaining catalogue exposure, longer training and conditional-uniform wrong-plane keys. It is not an isolated data-volume or architecture ablation. Real references are weak upstream affines; fixed 50um rendering PSF is an engineering assumption, not measured acquisition truth.

## Selected plane versus candidate-list capture

Physical capture@K means **some** retained candidate is within 10deg antipodal normal **and** 500um sign-aligned normal offset of the continuous reference. It does not mean that the model selected that candidate or recovered its finite image frame.

| Real six-donor development macro, all 64 images | Original A6000 | Whole C10000 parent | Final A22576 |
|---|---:|---:|---:|
| Selected normal error (deg) | 38.58545 | 12.42583 | **9.29486** |
| Selected normal-offset error (um) | 3348.90 | 955.52 | **763.85** |
| Physical capture@32 | 13.9394% | 90.6061% | **96.8182%** |
| Physical capture@128 | 26.3636% | 96.8182% | **98.4848%** |

Final selected finite-frame four-corner RMS, minimized over identity/horizontal correspondence, is **3,460.11um** (C: 3,352.56um). Better plane retrieval did not yield a better full-frame endpoint. Donor normal errors range **5.71019–16.88377deg**; capture@32 is 100% for four donors, 90.9091% for donor 15447 and 90% for donor 15935.

The pre-frozen image-free orientation prior, computed from equal-donor TRAIN normal scatter, scores **4.17220deg**, versus model **9.29486deg**: model minus prior **+5.12266deg**. It is better for five of six donors. This exposes the narrow real orientation distribution; the prior provides neither offset nor finite-frame localization and is not a calibrated posterior.

## Arbitrary-plane and TRAIN/development limits

Synthetic results use equal organizational-group means, **not biological-animal validation**. All 640 development rows are retained: 611 eligible and 29 censored. The near-zero-weight/censored rows are not relabelled successful localization.

| Synthetic development subset | Rows | Selected angle (deg) | Capture@32 | Capture@128 |
|---|---:|---:|---:|---:|
| All | 640 | 38.06257 | 76.4063% | 91.4063% |
| Eligible | 611 | **37.45970** | **77.8782%** | 92.1106% |
| Eligible smart-brush-absent | 207 | 41.08819 | 70.6250% | 88.8750% |
| Eligible smart-brush-accurate | 204 | 34.04375 | 84.1250% | 95.6667% |
| Eligible smart-brush-imperfect | 200 | 37.39901 | 79.2500% | 92.3333% |
| Censored | 29 | 54.40524 | 42.1569% | 75.4902% |

Eligible development normal-marginal MAP is also poor (**36.14379deg**), and exact-cell top1 is **4.69345%**. Versus original A6000, overall eligible selected angle improves by 1.17323deg and capture@32 by 4.18887 percentage points. Absent/imperfect selected-angle errors instead worsen by **1.16098/0.64093deg**, within the fixed +2deg retention allowance. Versus immediate C, eligible capture@32 is slightly lower (78.1442% to 77.8782%). A gate pass is not uniform improvement.

The corresponding 4,843 eligible frozen TRAIN rows score **32.18508deg / 91.4722% capture@32**, versus development **37.45970deg / 77.8782%**. Even TRAIN selected orientation remains poor, while its beam retrieval is much stronger; neither pure overfitting nor solved representation is established. Real all-TRAIN macro is **7.09319deg / 97.7375%**, versus development **9.29486deg / 96.8182%**. All 1,280 real TRAIN rows remain in readouts; **47** have zero paired-anchor supervision because both-chart support eligibility is not met. The 1,233 eligible real TRAIN rows score 6.75933deg; the 47 ineligible rows score 15.81876deg with 7.759mm offset error.

## Exposure and frozen decision

This stage attempted the remaining **50,304 distinct generated cells**, with **50,035 eligible** after finite/visible support weighting; 269 were censored, retained and not redrawn. Including frozen-row labels, this stage has **52,421 distinct eligible positive cells**. The lineage's earlier 48,000 plus these 50,304 completes **98,304 attempted catalogue-cell exposures**, not 98,304 successfully supervised positives, continuous subcell coverage, independent brains or successful acquisition-domain coverage. All generated modes were prebalanced at 16,768 presentations each. Frozen TRAIN rows contributed another 50,304 presentations.

The 12 pass flags comprise real improvement versus original A6000 (>=5deg and>=0.10 capture@32), real retention versus B8000, and synthetic eligible overall/each-mode retention (<=2deg angle regression and<=0.02 capture regression). They are fixed development engineering criteria, not evidence that selected planes, native joint refinement, uncertainty or deployment are solved. The next whole-lineage stage must preserve these distinctions; no benchmark beating claim follows.

## Frozen receipts

The independent audit reconstructed DEV full-score normalization, stable top128 rankings, physical geometry, group/mode summaries and all fixed gates. It verified whole-parent model/optimizer/RNG bindings, the exact 12,576-update schedule, fresh-cell exposure and TRAIN orientation prior; **18 retrieval tensors changed and 51 nonretrieval tensors remained exact**. Maximum DEV component-to-cell score discrepancy was 3.15843e-6; all saved DEV ranking/gate checks passed.

TRAIN descriptor reconstruction is **bounded-roundoff, not bit-exact**: 11 real and 101 synthetic rows have top-order differences within the fixed 2e-5 score allowance; 73 synthetic truth-rank intervals and two marginal-normal intervals were reported. Maximum reconstructed saved-top128 score differences were 9.52598e-6 real and 1.17236e-5 synthetic. Saved-top geometry and summaries were checked without pretending ambiguous reconstructed ranks are exact. Atlas/acquired-image rendering or decoding, encoder re-embedding and intervening optimizer/backpropagation were not replayed; neither were complete per-step negative-pool ranks/masks/quantile selections. Their frozen schedules, candidate unions and selected-owner admissibility were checked. Fixed A6000/B8000 references remained their previously audited pinned summaries, not newly rerun predictions.

Output root: `I:/AnatomyTracker/runs/joint_v6_full_coverage_ranking_001/`. Operative source commit `2935584bd1e6a22d28109422e6bf86086ea50e84`. Listed files were hash-checked after process exit. Complete input/source/schedule bindings and raw component/cell scores are retained in the run; TRAIN full-gallery scores are reconstructible from saved query descriptors, unchanged endpoint gallery and priors rather than all materialized as a second large score archive.

| Artifact | SHA-256 |
|---|---|
| Run completion | `576b85c7bcdbeea9e361b15908e320fe0d838291d083c945d19ede2b1687625e` |
| A completion | `77d13a9d678022e843a91f8425e5482341df3b97cb079d1549ec0f454f948b07` |
| Whole A22576 checkpoint | `e22ff8e13a09b8c624c64518ca65a0ac9feda2c823dde30b6b8b5707995d52d8` |
| A22576 gallery | `5219117c897bf68fb6139b5bd482bfdd05f5c52e2b9a5a7296c9936d0f8d0b36` |
| Archived operative driver | `d5489882e9fc41aafa0a35a0b7042c6fbb1e4b22da8dd6c2a22534e1c50c595c` |
| Generated schedule | `66cbc29c5f4036200895935a9e8b35089f4eb2a7a1dd368c6eb91abee8fdc6c1` |
| Sampling schedule | `44aef86b714a37238b89110502176c373cd8d69d5e48d280c226adbafeac4998` |
| Synthetic DEV summary | `3f0cdca17c31c4e3a00cbc4a0b7be47b564ab4ea0fa4aba0fc2fca3bbb127539` |
| Real DEV summary | `8e0478aa4c035124b3e3c56a93e97015aa4dc639076ee0de148cc55151535186` |
| Synthetic TRAIN summary | `5230772d7110d80a4709adad2bf8f606e7bba93b337c958b9d61c19abfdd02e7` |
| Real TRAIN summary | `0422d62f157cc72e484b91b0bcca4b314492efc27e64848571b270edef2fc6fc` |
| Independent audit | `c75c4ff817e7603eca31c8086080aafdb7f0953b3fec58c3f3f2f7842cf2afb3` |
| Independent auditor source | `368e077abf2fce6290a0f41cdcf8ece2d39433cfae9ff30f452f94767ea86dbb` |
