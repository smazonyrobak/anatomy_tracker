# All-32 pose-branch capture 023 — frozen development result

The read-only 023 audit used the frozen 019 batch-18,000 model and the same 185 synthetic DEV sections from eight synthetic deformation identities. It measured rigid tissue-coordinate error for all 16 predicted pose modes under both reflection states, using known synthetic CCF coordinates only for evaluation. An independent verifier passed checkpoint, panel, source, row and summary hashes. No model was trained or public benchmark used.

| Candidate pool | Identity-equal best rigid error | Cases with any candidate <0.25 mm | <0.5 mm | <1.0 mm |
| --- | ---: | ---: | ---: | ---: |
| Image-prior top eight | 1.175 mm | 0% | 12.4% | 57.3% |
| All 32 predicted branches | 1.025 mm | 0% | 12.4% | 60.0% |

The other 24 branches improve the physical best in 17.3% of cases, reducing mean best error by 0.149 mm, but they never add a candidate within 0.25 mm and do not increase the fraction within 0.5 mm. The earlier best-eight *refined/mapped* oracle has a different endpoint and should not be numerically conflated with this rigid control. The conclusion is narrower and robust: truncating from 32 to eight is **not** the main reason near-exact atlas matching cannot work. The one-pass global predictor does not supply a candidate in the anatomical matching basin even when all of its modes are retained.

Together with 022, this redirects development from additional same-candidate scorers to global proposal capture. The atlas image at the exact synthetic CCF surface is strongly distinguishable, but the closest predicted surface is still about a millimetre away. A better pose model or a broader learned retrieval/search representation must be judged by physically near-truth candidate capture on these fixed development identities, then by final selected error and unchanged donor-level weak-label regression. More branches without better proposals, a larger beam alone or an uncalibrated matching score is not enough. No deployment or DeepSlice superiority claim follows from this diagnostic.
