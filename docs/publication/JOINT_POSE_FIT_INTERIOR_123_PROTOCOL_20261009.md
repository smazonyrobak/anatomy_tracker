# 123 TRAIN-only interior anatomical-fit diagnostic (predeclared)

## Scope and bindings

Freeze the 122 step-2000 checkpoint. Generate 128 eligible independent v3 synthetic
TRAIN sections from the same 64 TRAIN deformation bases, using subject RNG seed
`20261009123` and consecutive draw seeds beginning `20261009123000000`.
Independently choose one virtual subject per attempt; accept only the generator's
own eligibility flag. Do not deliberately pair/repeat backgrounds, plans, or
planes. Record every attempt's full generator provenance and verify that no
physical-section ID overlaps any 122 draw (accepted or rejected). Bind the
122 completion/config/draw/checkpoint hashes, current source hashes, protocol
hash, and generated-output hashes. Do not use DEV, final-test, expert-real data,
training updates, calibration, or a GUI.

## Blind primary stratum

For each accepted image, use the 122 inference rule: prior top-eight base and
top-six anchor branches, plus two diverse unused anchors; apply its global
matcher, then atlas-conditioned local map to all 16 **corrected** candidates.
No synthetic truth, valid-tissue mask, or fit score enters proposals, matcher,
or map. Only after the complete beam is fixed, use the observed affine-gauge
truth and synthetic valid pixels to label the lowest mean rigid-gauge-error
candidate. Abstain if that error exceeds 1.5 mm. Among other candidates whose
error exceeds the best by **more than** 1.0 mm and whose **mapped** atlas-support
binary masks have Dice at least 0.85 and absolute area-fraction gap at most
0.08 versus the best, choose the lowest-rigid-error wrong candidate (slot tie
break). Abstain if none qualify. This truth-based pair label is diagnostic only;
it is not a deployable selector.

Render both mapped candidate atlases with the true section's finite-thickness
PSF. At side 96, threshold rendered atlas support at 0.95, intersect the two
masks, erode by a 17×17 all-ones square, and abstain below 256 common pixels.
This same fixed pixel set is used for every paired score. Do **not** intersect
the proposed score's mask with oracle tissue validity, an inferred segmentation,
or an image-derived tissue mask.

Primary score: area-downsample channel-0 source image and each supported atlas
intensity. Subtract a 9×9 box mean from each, take 9×9 local cross-products and
variances, calculate `1 - abs(correlation)` with epsilon `1e-5`, clamp absolute
correlation to `[0,1]`, and average on the fixed common pixels. Lower is better.
The comparison is polarity-invariant and uses atlas intensity beyond support.
The raw-121 baseline uses the original 121 weighted 9×9 local covariance and
variance equations on source/atlas intensity, with the synthetic valid mask
allowed **only** inside its local windows. For a fair paired readout, average
its `1 - abs(correlation)` at exactly the same common center pixels; this
fixed-center aggregation differs from 121's original variable patch weighting.
Run both scores again with atlas intensity identically zero but unchanged
support/maps/mask. Report mapped displacement RMS and finite-difference local
strain on the common pixels descriptively, never as a scorer or pairing rule.

For each pair, the margin is `wrong loss - best loss`; a win requires margin
at least 0.01 (ties and smaller margins are not wins). The proposed score
passes this **exploratory** anatomical-evidence gate only if at least 32/128
blind pairs are scored, its win fraction is at least 0.75, and its win fraction
exceeds the raw-121 baseline by at least 0.10 on identical pairs. Zero-intensity
control margins must be below `1e-6` in absolute value. Report all abstention
counts, score/margin distributions, support matching, and strain even if this
gate fails. Passing would justify a separate blinded, real-data and no-fit
controlled study, not training with this score or model promotion.

## Separate positive-control stratum

On the same 128 accepted images, separately compare the exact observed affine
gauge with six fixed diagnostic translations of ±1.6 mm along each CCF axis.
Map/render these seven states with the frozen model. From wrong states with
rigid error at least 1.5 mm, support Dice at least 0.85, and area gap at most
0.08, select highest mapped-support Dice (slot tie break). Apply the same
common-interior and score rules, and report its abstentions and win fractions
separately. This truth-generated control tests whether the score can see a
nearby anatomical displacement when blind pair geometry is unhelpful. It is
never substituted for the blind primary gate and cannot be used at inference.
