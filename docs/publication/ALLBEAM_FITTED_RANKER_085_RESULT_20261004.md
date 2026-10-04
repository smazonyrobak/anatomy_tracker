# 085 fitted all-beam ranker: development result (4 October 2026)

The preregistered selector gate **failed**. At batch 1,000, the eight-plan-equal synthetic visible-tissue mapping error was **2.591 mm** for the fitted selector versus **2.535 mm** for the matched parent 059 selector. The trained score changed the selected branch in 108/246 cases, improving 125 cases, but the mean worsened by 0.055 mm. It is not a deployable alignment, and this result does not justify a longer joint pose/warp continuation of the same ranking recipe.

| Frozen checkpoint | Selected 256-grid tissue error, synthetic (mm) | Best of the *same* 14 branches at 96-grid (mm) | Six-donor weak-real five-point disagreement (mm) |
| --- | ---: | ---: | ---: |
| Matched parent 059 | 2.535 | 0.986 | 1.138 |
| 085 batch 0 | 2.522 | 0.992 | 1.242 |
| 085 batch 500 | 2.565 | 0.992 | 0.743 |
| 085 batch 1,000 | 2.591 | 0.992 | 0.743 |

The 0.25-mm selected-error gain requirement is false at both trained checkpoints. The truth-best-14 nonregression requirement passes (0.992 versus 0.986 mm), as expected with the pose and local mapper frozen. The held-out weak Allen affines on 64 images from six real development donors are **not** verified atlas truth; their improved disagreement cannot override the failed synthetic physical-error gate or establish arbitrary-plane performance.

The batch-1,000 synthetic result is mixed across the predefined appearance and tissue-support strata. Exact-black sections worsen from 2.322 to 2.429 mm; imperfect-brush sections improve from 2.557 to 2.510 mm; raw-background sections are nearly unchanged (2.722 to 2.721 mm). Lower-support sections improve from 3.537 to 3.333 mm, whereas the other sections worsen from 2.188 to 2.306 mm. Three of eight synthetic deformation plans improve; five worsen. This heterogeneity is a warning against promoting the apparent real weak-label gain.

The run used 1,000 accepted independently sampled arbitrary-plane TRAIN sections from 1,100 attempted draws, with 1,000 unique physical section IDs and 1,000 unique sampled plane descriptions. The accepted appearances were exact black 336, raw 329, and imperfect brush 335. All attempts were TRAIN-only; the accepted sections retain 64 synthetic animal/specimen/experiment lineages. No plane was deliberately reused to produce an artifact/background pair. The parent 059 and correspondence 083 checkpoints were internally trained from random initialization; no external pretrained weights or pseudolabels were used. Only the spatial summary/quality path and fitted selector were trained; pose and local mapper were frozen.

The independent evaluator ran to completion on the frozen 061 panel (246 eligible sections across eight synthetic deformation plans) and the separate six-donor real development set. It reused the exact frozen parent rows and recomputed the same 14-branch map. Its 930 raw rows were independently aggregated by synthetic plan: parent selected 2,535.446 µm, model selected 2,590.814 µm, parent truth-best-14 985.794 µm, and model truth-best-14 991.858 µm. The output receipt hashes for config, rows, and summary were independently rehashed and matched. The evaluator verified all training source/checkpoint/draw/log bindings and the frozen panel and parent receipts before scoring.

Frozen training: `I:/AnatomyTracker/runs/allbeam_fitted_ranker_085_pilot`; completed receipt SHA-256 `707bbe5ff24f506a0da7b69e21799d607d91d7e44085c55f71437d4cf8ec0a90`. Frozen evaluation: `I:/AnatomyTracker/runs/allbeam_fitted_ranker_085_development_eval`; raw-row SHA-256 `cc44ce1bd7e6959f39c59aacb314754b59f09c76b0a7cdaa7d2e131691491abd`, summary SHA-256 `0ff961f5ce397b622c6542ed0b5b85e771bd773ef1bc10be07cac8e6201ffa6a`. The run followed [the frozen 085 protocol](ALLBEAM_FITTED_RANKER_085_PROTOCOL_20261004.md).

Next, examine score versus actual physical error **for every existing candidate** on the frozen panel, including top-*k* regret and tissue-support strata, without inserting the true pose or looking at images again. If the fitted evidence cannot order physically near candidates, change the representation/pose-update mechanism rather than extending this selector's training or permitting deformation to hide a wrong plane. Calibration animals, final-test animals, public DeepSlice data, GUI replacement, and numerical electrode-region probabilities remain untouched.
