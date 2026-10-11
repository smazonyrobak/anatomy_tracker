# 173: why did the 172 anatomy fit score fail?

Use only the frozen 172 TRAIN panel and its 143 preselected unconditional
support-matched near/wrong candidate pairs. This is an in-panel mechanism
check, **not** a new validation panel or a model-selection gate. No weights
are trained, no beam is changed, and no public/DEV/animal data are used.

Render each pair's original rigid finite-thickness atlas planes using the
same recorded PSF. At the pair's identical three-cell-eroded common atlas
support sites, compare the downsampled observed image with each atlas image
using Pearson correlation after mean and variance normalization. Do this at
native 48-cell resolution and after fixed 3x3 and 7x7 box smoothing. Do not
select the best scale after seeing truth. Repeat at the predeclared 3x3 scale
on the subset of sites where 172's synthetic observed-to-CCF valid-target
area is >=0.8; this is an **oracle visibility diagnostic only**, forbidden
at inference. Record all site counts, per-pair correlations, correct/wrong
decisions and provenance. Verify that the recomputed common-site count equals
the frozen 172 pair record. The train-free comparison can distinguish an
unhelpful 168 pointwise descriptor from low photometric identifiability or
the absence of source-visibility selection. It cannot establish a usable
deformation model or the final prediction accuracy.
