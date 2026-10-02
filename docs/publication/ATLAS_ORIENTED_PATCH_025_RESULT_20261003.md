# Oriented 2D atlas-patch descriptor 025 — frozen oracle-plane result

The independent verifier passed: 3,000 training batches, 6,000 accepted fresh TRAIN sections (573 rejected by existing eligibility), 192,000 positive patch pairs, optimizer/RNG/checkpoint/source/draw hashes, no TRAIN–DEV synthetic identity overlap, and four evaluations on 185 sections from eight held-out synthetic-development deformation plans. Accepted appearances were 1,993 raw, 2,031 exact black, and 1,976 imperfect brush. The atlas patch is a normalized finite-thickness render through the **known synthetic CCF surface**, with the local deformation supplied by the generator; the 025 model itself was initialized from scratch and received no prior model weights or labels from another network.

| Training batch | DEV recall@1 within 0.5 mm | DEV recall@16 within 0.5 mm | Exact appended positive ranked first |
| ---: | ---: | ---: | ---: |
| 0 | 0.24% | 4.83% | 0% |
| 1,000 | 63.12% | 91.79% | 49.56% |
| 2,000 | 90.83% | 99.43% | 85.02% |
| 3,000 | **96.22%** | **99.91%** | **91.99%** |

Values are averaged equally across the eight synthetic identities. At batch 3,000, recall@16 within 0.5 mm is 99.88% on raw, 100% on exact-black, and 99.87% on imperfect-brush sections. The preregistered necessary gate (≥25% top-1, ≥70% top-16, no mode worse than half the overall recall) passed. Mean training contrastive loss fell from 3.443 over the first 500 batches to 0.514 over the final 500; within-batch 1-of-128 top-1 rose from 30.88% to 93.48%.

This is a **narrow positive result**: a 2D atlas patch at the correct plane is distinguishable from nearby wrong patches despite the modeled synthetic contrast/background/artifacts. It explains why the 024 2D-to-one-3D-cube representation was a poor choice, and supports the [SLIV-Reg](https://arxiv.org/html/2410.18683v1) principle of extracting oriented 2D atlas patches. It does **not** show that an unknown plane can be found. The evaluation bank was truth-derived and contained an exact positive for every query. Even the held-out subjects are synthetic deformations of the same Allen atlas; real histology contrast, anatomical differences, tissue folds and acquisition artifacts may be harder. The next required check removes the oracle surface and measures matching on the true *rigid* plane and on the frozen 019 model's imperfect proposals before attempting a large multi-orientation atlas search or joint training. No numerical uncertainty, final-test-animal result, GUI replacement or DeepSlice superiority is established.
