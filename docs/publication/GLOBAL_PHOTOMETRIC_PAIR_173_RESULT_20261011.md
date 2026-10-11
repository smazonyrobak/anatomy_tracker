# 173: exact synthetic plane has image information; current proposals miss it

The [predeclared in-panel TRAIN mechanism check](GLOBAL_PHOTOMETRIC_PAIR_173_PROTOCOL_20261011.md)
re-rendered the same 143 unconditional, support-matched 172 pairs. No weights
changed and this is **not** an independent validation set. Pearson correlation
of the observed image with the correct candidate's rigid atlas render beat the
wrong candidate in 60/143 pairs at 48-cell resolution, 63/143 after 3x3 box
smoothing, and 73/143 after 7x7 smoothing. At the 3x3 scale, selecting only
synthetic known-valid image sites—an oracle visibility diagnostic forbidden in
inference—gave 62/143. Thus ordinary signed photometric correlation does not
provide a reliable broad-basin selector at the model's proposed poses.

The sampler deliberately inverts tissue contrast on about half its independent
draws. A post hoc analysis of the frozen pair correlations (not a predeclared
gate) found that the feasible sign-invariant absolute 3x3 correlation chose the
better proposed plane in 86/143 pairs. Correcting the sign with the *hidden
synthetic inversion flag* gave 90/143 and is only an oracle diagnostic. Neither
is sufficient blind evidence.

A second post hoc diagnostic rendered the **known true synthetic plane** at
the same finite-thickness PSF and shared comparison sites. Sign-invariant 3x3
correlation favored that true-plane render over the wrong proposed plane in
**137/143** pairs and over the beam's best proposed plane in **138/143**.
With oracle known-valid visibility, those counts were 141/143 and 141/143.
By contrast the better of the two *proposed* planes won only 86/143 using
the same sign-invariant score. The exact plane is supplied by generator truth;
it is not a model output. This is optimistic same-atlas synthetic evidence,
not a real-animal or physically acquired oblique result.

Taken together with 172's 0.939-mm truth-chosen best-of-16 rigid error, the
data suggest a narrow useful basin: atlas appearance is distinctive when the
plane is known exactly, but currently proposed planes are not precise enough
for this local comparison to select them reliably. This does not prove that
pose error is the sole cause: synthetic deformation, damage, background,
polarity and the learned descriptor/warp can contribute. The next main model
change should train a multiscale, atlas-conditioned full-pose corrector on
actual blind proposals, then train a constrained tissue map and test whether
its fit loss improves the same coordinate predictor. Do not promote the 172
score, call the 173 oracle a predictor, or benchmark DeepSlice.

Post-exit hashes of 173's config, raw rows and summary matched its receipt;
source, protocol and frozen 172 panel/receipt hashes also matched. All 143
records have distinct physical-section IDs and the script asserted identical
common-site counts against 172. Frozen result:
`I:/AnatomyTracker/runs/global_photometric_pair_173_train_001`, with
`completed.json` SHA-256
`7d4846c56a5fd3a1367dfa66b16302628121299f2d37875c35d78eb2896a733c`.
