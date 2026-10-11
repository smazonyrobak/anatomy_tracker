# 245: fixed-beam candidate-score diagnosis after the failed 244 readout

This is a diagnostic replay, **not training, checkpoint selection, calibration,
or a public benchmark**. The decision to perform it follows the frozen
[244 result](SHARED_ANCHOR_JOINT_FEEDBACK_244_RESULT_20261012.md). Keep all
code, inputs, temporary files and outputs on I:. Do not open CAL/FINAL data.

Use the unchanged 32 eligible synthetic DEV sections (four from each of the
eight synthetic plans) and the frozen 244 checkpoints at batches 1,000,
3,000 and 6,000. Run a matched pair on each section: (A) the 244 readout's
256-pixel input, 64-pixel first fit, 96-pixel scored fit, top-eight blind
prior beam; (B) downsample the *same* input to 192 pixels, use a 48-pixel
first fit, 64-pixel scored fit and top-three blind beam, matching the training
feedback geometry as closely as the frozen panel permits. No truth-guided
branch may enter either blind beam. The downsampled B case is an acquisition-
resolution control, not a new independent section.

For every blind candidate persist its branch ID, prior log score, fitted
score, fitted score excluding the prior, pose-refiner `delta` score, rigid
surviving-tissue physical error, mapped surviving-tissue physical error and
local-deformation RMS. Rescore the **same refined state and same mapped
coordinates** against the shifted atlas to isolate the learned matcher's
intensity dependence. The existing 244 end-to-end atlas-null rerun remains a
different counterfactual and must not be confused with this score-only null.
Count distinct pose modes in each beam, because opposite-reflection branches
can consume two slots, and report the spread of prior, learned-match and
deformation-penalty score terms separately.

Primary descriptive comparisons are identity-equal mean selected physical
error for prior versus fitted versus score-only null; best-in-beam oracle
regret; and within-section pairwise score/error concordance for candidates
whose rigid errors differ by at least 0.25 mm. Report these for total fitted
score and fitted score excluding the prior, at A and B geometry, and by
checkpoint. Also report how often the physical best-in-beam is selected and
the per-case score-only null change. A useful atlas-specific fit signal would
improve blind selection over the prior **and** predict lower physical error
more reliably than both the prior and score-only null across checkpoints.
There is no post-hoc threshold that can qualify 244 or justify scaling;
non-improvement directs us to a concrete score/geometry correction or a
different training objective.

The 244 feedback loss compares three blind candidates with a synthetic
truth/guided pair and at training resolution, whereas the 244 readout compares
eight blind candidates at larger resolution. These are hypotheses for the
failure, not established causes. This replay isolates resolution/beam and
atlas-specific scoring before any model change. Synthetic plans are not
biological animals, and these metrics cannot establish real heavy-oblique
accuracy or calibrated electrode-region probabilities.
