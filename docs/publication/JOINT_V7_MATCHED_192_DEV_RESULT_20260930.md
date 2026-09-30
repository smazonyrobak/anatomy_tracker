# Frozen 192px held-out synthetic diagnostic (2026-09-30)

The four disjoint synthetic development deformation subjects were re-rendered
at the model's actual 192px training resolution, from their previously frozen
physical planes and accepted 3D subject maps. This was **not** an enlargement
of the old 96px images. The preparation exited cleanly: 64 preselected physical
sections, three background/brush presentations each, 192 observations; 43 raw,
43 exact-black and 42 imperfect-brush observations met the predeclared tissue
support gate. Every 64 section-file digest, the records digest and protocol
digest matched the completed manifest. The frozen 30,000-step whole checkpoint
was then evaluated on all eight pose modes and both reflection branches for
every observation; every 192 raw-prediction digest and the two result-manifest
digests matched. No checkpoint was selected using these development subjects.

Subject-equal eligible readouts:

| Input | Direct MAP five-point error | Direct best-of-eight five-point error | Direct MAP normal error | Selected fitted centre-surface error | Fitted oracle branch error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw background | 8,197 µm | 3,380 µm | 38.45° | 3,195 µm | 1,318 µm |
| Exact black outside tissue | 8,667 µm | 3,329 µm | 40.30° | 3,471 µm | 1,369 µm |
| Imperfect brush | 8,735 µm | 3,670 µm | 38.23° | 3,220 µm | 1,429 µm |

The fitted error averages only visible tissue pixels, so it must **not** be
subtracted from the direct five-point error as if these were the same metric.
All four subjects contribute equally within each mode; ineligible observations
are retained in the raw outputs but excluded from these means. The results are
internal synthetic diagnostics on one atlas, not biological-animal validation
or calibrated probabilities.

The important architecture finding is that fitting changed the chosen branch
on **zero of the 128 eligible observations** (one changed branch was ineligible).
Inspection of the frozen v7 implementation confirms that its recurrent fitter
updates a deformation field but keeps the full pose state fixed; only a weak
training-time gradient from anatomical mismatch reaches the direct pose head.
This does not implement the requested inference loop in which fit quality
revises location/angle and the atlas is then re-extracted. Across the 128
eligible cases, selecting by the lowest fitting mismatch alone averaged
3,255 µm fitted surface error, versus 3,293 µm for the direct prior and 1,357
µm for the diagnostic oracle. Difficulty alone averaged 3,483 µm. These
values show that the current uncalibrated fit score does not recover the
available good hypotheses. They are diagnostic, not a tuned selection rule.

Decision: retain this entire frozen run as negative evidence. Add a
shared-weight full-pose residual update after each anatomical comparison,
re-render from the revised pose before the next comparison, and train candidate
quality/selection using known synthetic geometry. Only then spend the next
million-scale training budget; independent synthetic anatomy maps and the
TRAIN-only real cohort are being prepared separately. Do not benchmark this
model against DeepSlice or claim electrode-site region probabilities.
