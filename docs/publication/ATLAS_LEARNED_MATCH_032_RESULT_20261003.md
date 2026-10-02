# Contextual atlas-match selector 032 — frozen development result

The first 3,000-batch stage used 6,000 fresh synthetic TRAIN sections, with one randomly sampled appearance per physical plane and preserved draw identities. It continued the scratch-trained project descriptor 025 as a frozen front-end and initialized the set-wise selector randomly. The bank was atlas-only; no DEV examples entered training. An independent raw-output audit passed training/evaluation hashes, four checkpoint summaries and every saved affine's physical-error replay on the 185-section/eight-identity synthetic DEV panel.

| Selector batch | Correct chosen match among queries with an available correct top-16 candidate | False match among queries without one | Mean rigid error from weighted plane fit |
| ---: | ---: | ---: | ---: |
| 0 | 5.16% | 75.84% | 3.861 mm |
| 1,000 | 0% | 0.03% | 3.738 mm |
| 2,000 | 0.18% | 0.10% | 3.616 mm |
| 3,000 | **0%** | **0%** | **3.585 mm** |

The preregistered necessary gate (≥50% conditional correct-match recall, ≤20% false matches and <2.5 mm fitted error) **failed**. The network learned to avoid false assignments by almost always choosing “no match”; it did not learn a useful plane. It was not added to inference. The median rank of the physically nearest available candidate among the 16 match logits improved from 9 to 5, but its median probability fell to 3.28%, while median no-match probability rose to 62.23%. Therefore the output is not just a random untrained selector: there is weak ranking learning that the all-17-way classification objective fails to turn into usable matches. More batches of the same loss are not justified by this evidence.

The targeted continuation 033 keeps the same architecture and optimizer lineage, but explicitly separates (i) ranking candidates *conditional on a true candidate existing*, (ii) deciding whether any match exists and (iii) supervising the differentiable plane fit using known synthetic physical coordinates. This addresses the observed no-match collapse and gives the pose path a direct learning signal. It still will not establish the requested atlas-fitting-to-pose feedback; that must be added and verified after pose capture becomes useful. The current run is synthetic DEV only, with oracle tissue-pixel query selection and uncalibrated scores.
