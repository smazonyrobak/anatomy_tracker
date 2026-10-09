# 112: first fitting-evidence result

The 512-batch TRAIN-only pilot completed from frozen 111. It used 1,024 distinct accepted full-angle synthetic sections (1,200 attempts), with exact, nearby and predicted hard-wrong atlas planes and a different-image contrast. The 16-section fixed synthetic DEV readout has two sections from each of eight independent deformation plans; its identities are disjoint from all TRAIN draws. Run source, protocol, parent, data, checkpoints, draw logs and evaluation rows/summary passed their bound SHA-256 checks. It is not a real-animal accuracy, calibration, GUI or public-benchmark result.

| Plan-equal DEV measure | Frozen step 0 | Step 128 | Step 512 |
| --- | ---: | ---: | ---: |
| Exact-plane rigid dense error, mm | 0.2473 | 0.2473 | 0.2473 |
| Exact-plane mapped dense error, mm | 0.2472 | **0.2261** | 0.2263 |
| Sections whose map beats rigid | 7/16 | 12/16 | 12/16 |
| Fitted score ranks exact above wrong | 10/16 | 15/16 | 15/16 |
| Fitted score ranks correct image above swapped image | 15/16 | 15/16 | 16/16 |
| Nearby score derivative points toward exact plane | 14/16 | 13/16 | 16/16 |

The scorer learned a useful **local** direction and the mapper improved modestly at a known correct plane. This is not yet proof that internal atlas anatomy drives the gain: an oracle surviving-tissue-mask/atlas-support Dice control ranked exact above the hard-wrong candidate on **16/16** even at step zero, versus the learned full score's 15/16 at step 512. Fifteen of the 16 wrong candidates matched the exact plane's *mean support fraction* within 0.10, but their **spatial silhouettes** were still distinguishable. The same oracle silhouette cue also ranked the fixed near plane above the farther plane on 16/16. The mask is not deployable and gives the control privileged truth, but it exposes a confound in this small panel. Score differences are uncalibrated logits, not anatomical probabilities.

The preregistered gate requiring added information beyond atlas support is therefore **unresolved/failed**, not a license to send this score into global pose probabilities. A matched TRAIN-only support-channel control, with the same initialization/draws/optimizer but the atlas intensity channel removed, is the next minimal experiment. Compare it to 112 on a wider fixed synthetic panel, especially the 48 sections not in this pilot readout; retain 112's 16-sections as exploratory rather than a fresh confirmation. The separate 113 result also shows that fit feedback cannot rescue a virtual cut when even the best coarse plane is several millimetres away.
