# Bounded normal-readout control, 2026-09-29

The unchanged proposal can memorize eight fixed observations in both FP32 and
the actual encoder-AMP/FP32-head configuration. This rules out a universally
non-learning path, not a population-level capacity limitation or every data bug.
Completed005's extra normal loss still leaves about42 degrees of coarse normal
error. The next control changes the orientation readout, not data or precision.

`proposal_normal_readout_count=384` enables a zero-initialized linear readout of
the existing64-dimensional image context. Its normal-only logits tilt the
**final mixture**. For component probabilities C_l and weights w_l, the code
computes Z_l=sum_x C_l(x)exp(b_normal(x)), replaces C_l with
C_l(x)exp(b_normal(x))/Z_l, and replaces w_l with normalized w_l Z_l.
Consequently the final probability is proportional to P(x)exp(b_normal(x)).
For fixed base logits, every within-normal offset/roll conditional is unchanged;
the returned component distributions and weights still reconstruct the result.
Training can of course change the shared encoder and original conditionals.
All98,304 cells remain represented. Probabilities remain uncalibrated.

DefaultNone preserves old parameters, RNG and predictions. The opt-in readout
is initialized inside an RNG fork so every pre-existing parameter initialization
is retained. A single completed003-image/full-catalogue GPU check found exact
default prediction equivalence and zero-readout equivalence; a nonzero normal
bias changed within-normal log-conditionals by at most1.53e-5 (FP32 roundoff).
Returned-mixture reconstruction was exact, gradients were finite/nonzero, and
permuting cells together with their IDs changed log probabilities by at most
2.29e-5. Check receipt: `I:/AnatomyTracker/tmp/check_normal_readout_tilt_20260929.json`.

Run006 starts a fresh complete model with the same seed2026092805, data,
appearance, cell order, frozen-row order, optimizer and joint+normal objective
as005. It runs only4,000 updates. The driver explicitly retains a20,000-update
**schedule-generation horizon** so shortening execution cannot silently change
the RNG-dependent schedule prefix. Only organizational ID namespace differs.
No005 weights, earlier-project weights, external features or pseudolabels load.

Predeclared engineering decision at the fixed4k endpoint: retain a head-only
long-run priority if marginal-normal NLL improves by at least0.25 and mean
marginal-normal angle by at least5 degrees versus005 at4k, with all brush modes
reported. Otherwise prioritize the deeper shared spatial encoder next rather
than spending another20k steps on this head alone. These thresholds prioritize
experiments; they are not biological accuracy, significance or release gates.
Native joint pose/deformation learning remains required either way. No public
benchmark, final-test exposure or calibration claim is involved.
