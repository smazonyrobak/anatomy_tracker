# 322: internal anatomical boundaries do not rescue C4 plane selection

The frozen 321 model, its 128 TRAIN-inner synthetic cases, and their top-16 candidate planes were scored without training or changing any checkpoint. The [predeclared protocol](C4_INTERNAL_BOUNDARY_322_PROTOCOL_20261015.md) compared rendered internal Allen-class boundaries with source-image edges, against exterior-support overlap and a shuffled-boundary control. It even used the **true synthetic surviving-tissue mask** to exclude exterior edges. That mask is unavailable at inference, making this an optimistic information test rather than a deployable method.

Boundary evidence was available for a mean 15.05/16 candidates; the first 16/16 cases passed both observability checks. It was therefore meaningful to score all 128 cases. Yet the proposed cue failed six of eight information-gate clauses.

| Selector on the same 128 cuts | First-choice mean rigid-site error | First-choice >5 mm |
| --- | ---: | ---: |
| Frozen 309 | 2.513 mm | 10 |
| Frozen C4 prior | 2.575 mm | 13 |
| Internal-boundary cue | 2.711 mm | 9 |
| Atlas exterior-support overlap | 2.594 mm | 11 |
| Shuffled-boundary control | 2.462 mm | 10 |
| Truth-best of C4 top 16 (unavailable oracle) | 0.831 mm | — |

For 115 support-matched near/far candidate pairs, the boundary cue preferred the near plane 54.8% of the time, versus 60.9% for exterior support and 44.3% for shuffled boundaries. Only three of eight synthetic base-animal means were non-worse than the C4 prior; the protocol required six. The lower >5-mm tail for boundary scoring does not compensate for its worse mean and weak pair discrimination. The shuffled control's apparent mean gain is further evidence that incidental reordering of these candidates can look helpful without a trustworthy anatomical signal.

Decision: **do not train a selector using this boundary cue**, and do not promote a model, claim real-animal accuracy, or run a public benchmark from this result. The repeated negative selector experiments (079, 125, 128, 299, 300, 304, 305, 322) justify stopping further minor atlas-score variants; they do not rule out a substantially different image/atlas feedback architecture. Next work should test whether realistic, optional acquisition metadata actually resolves the frozen candidate ambiguity, while independently measured physical oblique-plane validation is sought. If neither supplies enough information, report that limit instead of forcing a confident arbitrary-plane estimate.

The completed CPU-only output is `I:/AnatomyTracker/runs/c4_internal_boundary_322_002`; its 128 rows and config/rows/summary receipt hashes passed a post-run audit. The `_001` attempt stopped at summary serialization after scoring; it remains preserved and incomplete, with identical scored-row hash. No DEV/CAL/FINAL animals or public benchmark were used.
