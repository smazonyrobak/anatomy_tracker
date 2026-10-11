# 179 TRAIN-only audit: is the synthetic rigid-pose label fitted to empty canvas?

The 166 generator obtains its atlas-plane `state` by an affine fit to the
entire warped 256×256 coordinate field before image appearance or defects are
added. Much of that square is outside visible brain. A model cannot infer
subject deformation in empty canvas, so this audit measures whether the
reference plane differs materially from a fit to atlas-supported tissue.
It does not train, select, or modify a model.

Freeze the 166 generator and 64 TRAIN deformation bases. Draw two independent
full-sphere, support-intersecting physical sections per base with separate
seeds (128 total); do not reuse one physical plane with a different
background. Work on the pre-artifact `centre` coordinate field and
finite-thickness atlas support, so tears/background/brush do not themselves
move the label. Fit one affine atlas surface by weighted least squares at
pixels with support >0.25, weighting by support. Correct the chart's x axis
for the section's recorded reflection. Compare its predicted CCF position
with the generator's original whole-canvas `state` on those same tissue sites.
Record the original and tissue-fitted residual to the coordinate field,
their mean tissue-site positional separation, normal-angle separation,
five-frame-point separation, tissue fraction and fit condition number.
Retain all draws and strata; report sparse lowest-quartile and ≥30°
nearest-cardinal cuts separately.

Predeclared interpretation: median tissue-site pose separation ≥0.25 mm or
upper-quartile separation ≥0.50 mm is large enough to warrant changing the
pose label/loss before another architecture run; smaller differences mean
this particular label convention is unlikely to explain multi-millimetre
blind errors. The tissue fit is an alternative synthetic gauge, **not**
expert truth. It may absorb biological deformation that should instead go
into the local map, so any training change must preserve physically
identifiable pose/deformation roles and check real-animal outcomes later.
No DEV, final animal, real expert label, GUI, probability calibration, or
public benchmark is used.
