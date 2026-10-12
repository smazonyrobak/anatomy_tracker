# Candidate 299 — candidate-aligned atlas ranker did not pass

## Frozen scientific result

The precommitted source and [protocol](ORIENTED_SLAB_RANKER_299_PROTOCOL_20261012.md) were run once from the frozen 275 step-8000 pose checkpoint on I:. The independent verifier reloaded both final rankers, reproduced all 64 saved 16-way scores, checked all source/protocol/input/output/checkpoint SHA-256 digests, TRAIN base separation and physical-section IDs, and reaggregated the raw decision. The runner exited successfully; `completed.json` SHA-256 is `d67d79c0fe5e112bfe7d63b9373c418d041d8bf098c7737b77c93197fdf51903`.

| 64 new TRAIN-inner sections, bases 56–63 | Mean rigid-site error | Support-matched near/far wins (49 pairs) |
|---|---:|---:|
| Frozen prior first choice | 2.6306 mm | — |
| Fresh ranker, atlas intensity + support | 2.6320 mm | 29/49 (59.2%) |
| Matched fresh ranker, support only | 2.6594 mm | 28/49 (57.1%) |
| Best of frozen blind top 16 (truth oracle, not selectable) | 1.2113 mm | — |

Only the pair-count clause passed; the other five predeclared clauses failed. Atlas intensity improved selection by merely 0.0274 mm over its support-only control and worsened it by 0.0015 mm relative to the frozen first choice. The full ranker changed 12/64 first choices: five improved and seven worsened. Pair preference differed between the two arms in only 3/49 eligible sections. The first and last 200 training-presentation mean pair losses were 2.712→2.474 (full) and 2.711→2.467 (support-only): optimization occurred, but no convincing incremental anatomical ranking signal emerged. Do not scale, promote, or integrate this unchanged ranker.

The brush/no-brush descriptive means were respectively 2.764→2.833 mm (18) and 2.578→2.554 mm (46) for frozen prior→full ranker. Seven sampled sections met ≥45° obliquity; their 1.989→1.982 mm change is too small and too sparse to claim oblique competence. These are synthetic TRAIN-inner results, not acquired-slice error, not animal-separated biological validation, and not calibrated electrode probabilities.

## Interpretation and next decision

The top-16 oracle gap remains substantial, but support-only and full scores are almost interchangeable. This is evidence against simply appending this candidate-level oriented-slab score or training it longer unchanged. It does **not** establish that atlas anatomy lacks information, that any-plane inference is impossible, or that a cardinal-plane restriction is warranted. The ranker had only candidate-level supervision; it was not taught which individual source points correspond to which atlas points. A next bounded, predeclared experiment can exploit exact synthetic point correspondence while retaining candidate-aligned, fixed-denominator shared-residual scoring and the matched support-only arm. It must demonstrate an atlas-intensity advantage and first-choice gain before any further scale-up. Independently ground-truthed physical oblique histology and held-out animal calibration remain missing for the eventual deliverable.

No DEV/CAL/FINAL animals, public benchmark examples, external trained weights/features, or pseudolabels were used. The 275 proposer and all its 16-way beams remained frozen. No local tissue-to-atlas warp or GUI code was altered.
