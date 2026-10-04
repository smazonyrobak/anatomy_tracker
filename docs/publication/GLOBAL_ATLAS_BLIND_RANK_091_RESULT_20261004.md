# 091 blind-pair atlas ranking: negative result

The predeclared 30,000-section TRAIN continuation and frozen 246-section synthetic DEV readout completed. The 091 scorer is **rejected**: it removed most of 090's atlas-support shortcut, but did not learn reliable anatomical ordering and worsened selected location accuracy. Do not promote it to the GUI or use its scores as calibrated probabilities.

| Frozen DEV, equal weight to eight synthetic plans | 089 baseline | 090 final / 091 step 0 | 091 step 5,000 | 091 step 15,000 | 091 step 30,000 |
|---|---:|---:|---:|---:|---:|
| Selected mapped 3-D CCF error, mm | 2.592 | 2.828 | 2.939 | 3.047 | **2.992** |
| Best available of 14, mm | 0.961 | 0.961 | 0.961 | 0.961 | 0.961 |
| Score / lower physical error Spearman | — | 0.044 | 0.055 | 0.084 | **0.096** |
| Score / atlas support Spearman | — | −0.845 | −0.076 | −0.308 | **0.027** |
| Top-three best available, mm | — | 1.650 | 1.790 | 1.675 | 1.811 |

The final 091 selection is 0.400 mm worse than 089 and 0.163 mm worse than 090. Six of eight synthetic plans worsen relative to each comparator. Exact-black selected error is 2.771 mm versus the 089 baseline 2.562 mm, exceeding the predeclared +0.150 mm regression allowance. Raw-background error is 3.276 versus 2.630 mm; imperfect-brush error is 2.893 versus 2.430 mm; low-support error is 4.425 versus 3.220 mm. All three prespecified gates fail: ≥0.400 mm improvement, ≥0.30 within-section score/error rank association, and ≤0.150 mm exact-black regression.

Training did optimize the intended problem: the first and last 1,000 accepted sections had mean pairwise loss 1.287 and 1.249, with 32.45 and 32.77 support-matched pairs per section and no zero-pair sections in those windows. Nevertheless, the held-out candidate ranking is weak. This is evidence against simply adding more batches or another support penalty to this global image/atlas embedding. A more spatially explicit correspondence/fitting representation is the next targeted change; its fit quality must affect the direct probabilistic pose predictor during joint training, not merely rerank a frozen beam afterward. That is a project-specific inference, not a general conclusion about contrastive models.

The independent evaluator asserted exact 090 step-0 score/support parity, the same 086 candidate IDs and 089 corrected states/errors, frozen-source and output hashes, section IDs and provenance, and 30,000 accepted TRAIN draws. A separate raw-row audit found 13,776 candidate rows, 984 section/checkpoint groups of 14 with exactly one score-argmax selection and true minimum marked in every group; recomputed final plan-equal selected error was 2,991.577565 µm. No output tree was read or modified during the live runner. This panel is synthetic and derived from one atlas; its eight plans are not independent biological animals and do not establish real histology accuracy, uncertainty calibration, or DeepSlice performance.

Frozen artifacts on I: `runs/global_atlas_blind_rank_091_pilot` and `runs/global_atlas_blind_rank_091_development_eval`. Protocol SHA-256 `9657708cfba91b683072639230712ffbac3565234314d70fb1d948fccae3cf99`; TRAIN completion SHA-256 `8c65b6b53be570afc8cb18c013a56f27ae1c9e86ff05a69f4311f88e47f39b52`, final scorer checkpoint SHA-256 `63e1f021393811f39981b69d421021e5e938edd7b2c635a725a196b99b5110c1`; DEV completion records `sections_sha256=aea98bb21b3e42db6639f1303d339eb6e5ddabb227c620dce4b9ce0620ceb7d1` and `summary_sha256=c1e0ef6e031166bc48551d77f9898dd783d71ef06cf5bcab75844d708934d04b`.
