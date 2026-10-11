# 172: frozen-map coherent fit-energy assay (TRAIN only)

The 171 hard-beam matcher failed its frozen selection gate. Its score can pick
independent local correlation peaks without requiring one physically coherent
deformation. This assay tests whether the already trained, affine-gauge-separated
132 local map provides a useful *measured fitting difficulty* when evaluated at
the same blind planes. It adds no trainable parameters and cannot by itself
establish the optimum achievable warp.

Freeze 192 unconditional and 48 orientation-conditioned 45–55° TRAIN attempts
from bases 60–63, using fresh seeds independent of 169–171. Retain blanks;
sample physical geometry and v4 appearance independently on each attempt.
Generate the 132/121 blind 16-branch beam, reflection, direct prior and native
fit before reading any section truth. Keep true synthetic PSF offsets/weights
fixed for this diagnostic, noting that deployment must estimate them. Every
branch uses frozen 132 `map` once with real atlas intensity and once with
zeroed atlas intensity but unchanged atlas support. Render the finite-thickness
atlas at the mapped coordinates on a 48-cell grid. The image-only 168 source
and atlas descriptors supply pointwise cosine agreement. Do not use synthetic
`valid_mask`, truth, animal ID, or automatic segmentation in any score.

For branch c and comparison sites M, define its appearance term as the mean
of `S_c*(1-cosine)+(1-S_c)`, where S_c is mapped atlas support. Add 0.05 times
mean squared local displacement divided by 0.5 mm and 0.05 times mean squared
physical in-plane displacement gradient divided by 0.2. All terms share the
same M. For blind 16-way ranking M is each branch's eroded pre-map atlas
support; add 0.2 times the missing pre-map support fraction of the full raster.
For a matched-pair decision M is the identical three-cell-eroded intersection
of the two pre-map supports, at least 64 cells. No labeled threshold, scale,
or checkpoint is tuned on the panel. Blend with the direct prior by subtracting
one within-beam standard deviation of energy from one within-beam standard
deviation of prior log mass; report energy-only and prior-only separately.

After all inference scores are fixed, truth chooses the lowest-rigid-error
branch at <=1.5 mm and the highest-prior branch >=1.0 mm worse that has
pre-map atlas-support Dice >=0.85 and area-fraction gap <=0.08. The pair is
matched on common sites and is never used for a blind claim. Compare coherent
energy with rigid-no-warp, 171 local-peak score, score-stage zero-intensity,
and the complete zero-intensity mapping-and-scoring rerun. Record mapped
support, displacement, strain, physical derivative bound, removed affine
coefficients, paired ranks, all 16 branch energies, selected slots, rigid and
fitted mapping errors, exact provenance, source/checkpoint/panel hashes.
Also report an atlas-support-only blind selector (largest pre-map supported
fraction), since earlier fit scores sometimes lost to this shortcut.

The *pair gate* requires at least 40 qualifying unconditional pairs; >=75%
correct lower-error decisions; and >=10 percentage-point advantage over both
rigid and full atlas-intensity-null energy. If it fails, do not promote this
energy into pose training. If it passes, the *blind gate* requires its fused
selector on nonblank unconditional attempts to improve initial rigid error by
>=0.25 mm and capture <=1.5 mm by >=10 percentage points versus both the
direct prior and native 121 fitter, while its selected map error must not
regress against native fitted error. Also require a >=0.15-mm / >=5-point
advantage versus fused full intensity-null, and >=0.15-mm / >=5-point
advantage versus support-only selection. Report sparse and oblique strata
without pooling them into unconditional population performance. Both gates
must pass before testing feedback gradients; even success would justify only
the next synthetic stage, not real-animal claims, calibrated probabilities,
GUI deployment, or DeepSlice benchmarking.

104/105 already failed with older 094-based fit fields. If 172 fails, inspect
whether its frozen 132 warp is itself inaccurate at near branches before
inferring that coherent fit is uninformative. A bounded per-case optimized
smooth field with held-out-site scoring would be the next discriminating
experiment, not a broad new network or more scale of 171.
