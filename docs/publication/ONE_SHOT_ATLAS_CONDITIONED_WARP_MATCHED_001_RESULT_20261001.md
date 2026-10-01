# One-pass atlas-conditioned fitter: matched development result

The atlas-conditioned fitter is promising on fresh, strongly warped synthetic sections **from training deformation bases**, but it has not passed held-out-deformation development. It is not a usable joint alignment model yet: this experiment supplied the correct plane during fitter training and froze the plane predictor, so fitting did not teach plane prediction.

## Frozen comparison

Both arms started from the same randomly initialized-lineage mixed-real step-20,000 checkpoint. Each received the same 12,000 accepted, distinct synthetic section draws over 4,000 batches (three sections per batch); 13,221 attempts were logged. The image-only arm continued its existing local fitter. The conditioned arm additionally saw a finite-thickness atlas rendering at the supplied plane and learned image–atlas matching features. The common image encoder and pose head were frozen, and each fitter used a fresh optimizer. No public benchmark, calibration animal, or final-test animal was used.

The pre-existing development panel has 64 physical sections from four disjoint synthetic deformation plans; one preselected appearance per section was read at native 192 pixels and resized to 256. Forty-three were eligible. Animal-equal, visible-tissue point-mapping errors at the **correct supplied plane** were:

| Fitter | Step 0 | Step 2,000 | Step 4,000 |
|---|---:|---:|---:|
| Rigid plane, no local warp | 58.98 µm | 58.98 µm | 58.98 µm |
| Image only | 59.09 µm | 58.88 µm | 58.84 µm |
| Atlas conditioned | 59.09 µm | 60.46 µm | 60.75 µm |

The conditioned fitter therefore **regressed by 1.77 µm** versus the rigid plane on this older panel. At the same selected predicted plane, its 1.11 µm improvement was negligible against 4.41 mm total mapping error. This panel predates the stronger local-warp augmentation used for current training and has a much lower rigid error; the comparison cannot establish whether the fitter generalizes to held-out subjects under the current warp distribution.

To check that distribution mismatch, a separate fixed-seed diagnostic drew 128 distinct eligible 256-pixel planes across 57 of the 64 **TRAIN synthetic deformation bases**, with a seed disjoint from training draws. It kept the first eligible draws rather than selecting by outcome. Draw-mean correct-plane error was 142.16 µm rigid, 142.15 µm image-only step 4,000, 138.45 µm conditioned step 2,000, and **128.43 µm conditioned step 4,000**. The 13.73 µm conditioned gain versus rigid appeared on 46/57 sampled bases; animal-equal gain was 12.54 µm. Gains were 1.10, 15.10, and 25.03 µm over increasing warp-strength thirds, and 12.90, 14.17, and 13.78 µm on raw, exact-black, and imperfect-brush appearances respectively. At a fixed predicted plane, conditioned step 4,000 gained only 1.95 µm against 3.70 mm rigid error. These training-base draws are not independent animals or held-out validation.

The training log is consistent with learning a nonzero local field: mean mapping error over batches 3,901–4,000 was 121.93 µm for the conditioned arm versus 135.09 µm image-only; mean field RMS was 18.02 versus 0.81 µm. This supports a real in-distribution fitting effect, not successful global plane capture.

## Decision and next gate

Do not promote either fitter or scale the architecture from this result alone. Render a native-256, one-appearance-per-plane panel from the four frozen held-out synthetic deformation plans, apply the same current one-shot distortions, and compare the same frozen checkpoints. If that retains a material gain, continue joint training so the fit objective actually changes the pose head, while separately improving the large global pose error. If the gain disappears, diagnose domain mismatch or fitter overfitting before further training. Neither synthetic panel substitutes for animal-held-out expert histology or final calibration.

Raw matched run: `I:/AnatomyTracker/runs/one_shot_atlas_conditioned_warp_matched_001/`. Its `config.json` SHA-256 is `c9de2c1d57536d4cd2690249444fe8525d4db110cfbff846b29e2e4dff98813a`; shared draw log SHA-256 is `379209b154504dfc7aec81e6b1b13ab320b70d96ef567d1232e1d329b4102c61`. Each arm has 4,000 training rows, 192 development rows over three checkpoints, and a completion receipt. The 4,000-step checkpoint SHA-256 values are `f1a7511a2082cff9b134fa677bca3650ba57f00a33abb2fdfb51779286c7adfc` (image-only) and `c60bc6216a0481b7f7b5198c7b0c7425b9ea5b986875251501fee4378ff6a0b6` (conditioned).

Fresh diagnostic: `I:/AnatomyTracker/runs/one_shot_atlas_conditioned_warp_fresh256_audit_001/`. Its 128 provenance-complete rows and 140 attempts have SHA-256 `6ee149fe0993f136515d537800c99def609f885577045e375635a6758d29a03a` and `be69915091c371b25bf3f1aa6f81fa02459c6f68151b1de03443d7d2ebae2405`; full metrics, source/checkpoint hashes, and per-base results are in `summary.json` (SHA-256 `f0c862ba81afd49852b804c57e0b31418b6394e98a8ca97566858c26689142e2`) and `READOUT.md`.
