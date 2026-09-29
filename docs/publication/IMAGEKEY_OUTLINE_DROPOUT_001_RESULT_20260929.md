# Optional-outline dropout001 — audited negative result

The matched comparison finished all2,000 additional updates per arm, starting
from the same complete image-key001 step4,000 model and AdamW state. Terminal65125
exited0 after1300.38s, source `14aa9fa`. A keeps optional metadata; B removes
boundary/availability on the frozen50% draw for available training observations,
without changing image pixels, targets, examples or negatives.

## Fixed-gate result

| Endpoint | Original4k | Control A6k | Dropout B6k |
|---|---:|---:|---:|
| Real6-donor macro normal error, degrees |45.05|38.59|41.75|
| Real normal-offset error, um |4150.58|3348.90|3057.13|
| Real top32 plane capture |9.09%|13.94%|15.30%|
| Real top128 plane capture |21.36%|26.36%|29.24%|
| Eligible synthetic macro normal error, degrees |40.36|38.63|38.45|
| Eligible synthetic top32 plane capture |68.46%|73.69%|73.85%|

Capture requires one candidate with antipodal normal error<=10deg and signed
normal-offset error<=500um. Real results cover unchanged64 images from six
development donors, against weak upstream Allen affine references. Synthetic
results cover611 eligible observations in40 organizational groups, not40
biological animals. All640 synthetic rows, including29 censored rows, are saved.

**The dropout gate fails.** B worsens real mean orientation by3.168deg rather
than improving it by at least5deg, and improves top32 capture by only0.013636
rather than the required0.10. All synthetic retention conditions against both
A6k and the original4k parent pass, overall and by original input mode.
All six donor-specific orientation means worsen with dropout.

Descriptive paired-donor bootstrap effects, B minus A (10,000 resamples of six
whole donors with replacement, NumPy PCG64 seed2026092919, percentile95% CI):

- Normal error: +3.168deg, CI[+1.852,+4.845].
- Top32 capture: +1.364 percentage points, CI[-5.000,+7.576].
- Normal-offset error: -291.77um, CI[-669.40,+37.99].

These small-development-cohort intervals are descriptive, were not promotion
criteria, and do not replace independent final validation. They do not measure
uncertainty calibration of model predictions.

## Independent post-exit audit

Terminal44079 exited0. The audit authenticates original input/source bindings,
regenerates both RNG schedules and effective dropout masks, compares every
paired trace entry and first-batch tensors, checks complete optimizer budgets
and frozen non-retrieval tensors, and recomputes ranking/physical-plane errors
with independent NumPy geometry. Component-to-cell full-score normalization
differs by at most3.5273e-6. All numerical summaries and fixed gate decisions
agree. Integrity passes; scientific dropout improvement does not.

Limits: no model descriptor re-embedding or score reconstruction from descriptors;
no independent replay of every nearest-negative pool. Initial in-memory state
copying is authenticated by source assertions and receipts, not re-observed.
This is not a topology, biological calibration or public benchmark audit.

Frozen paths under `I:/AnatomyTracker/runs/`:

- `joint_v6_imagekey_outline_dropout_001/` — both complete endpoints, full scores,
  gallery/query descriptors, checkpoints/optimizer/RNG, schedules and identities.
- Completion SHA-256:
  `515d583cabd70ef54b7cde9a739165c44c5d407a2341947eea595e1abad8f0b1`.
- `joint_v6_imagekey_outline_dropout_001_independent_audit/audit.json` SHA-256:
  `2db902bdc6a2e502c911fa3ced308ed96d27219926356457efa0e063d53bd2e0`.

CUDA section mapping overlapped part of training; elapsed time is not a hardware
benchmark. All artifacts and caches remain on I:.

## Decision

Do not promote B or tune more dropout probabilities against these six donors.
Optional smart-brush inputs remain part of the required model design; this
specific omission control does not establish their optimal handling. It does
show that removing the metadata association alone is not a demonstrated remedy
for the current real-domain failure. Proceed to the prepared weak-affine real
training plus arbitrary-plane synthetic replay. The separate native curved-ribbon
control keeps its predeclared original4k whole parent. No checkpoint is a
qualified joint model, and neither38.59deg nor41.75deg is usable alignment accuracy.

Use whole A6k as the next coarse-training warm start: its eligible synthetic
orientation and top32 capture improve in every original mode relative to4k.
This is not a qualified-model promotion. Real donor15219 worsens from41.01deg
to53.58deg, three donors still have zero top32 capture, and the donor-macro
capture gain of0.048485 does not meet a0.10 improvement criterion. Keep these
failures visible; no public benchmark or confidence claim is justified.
