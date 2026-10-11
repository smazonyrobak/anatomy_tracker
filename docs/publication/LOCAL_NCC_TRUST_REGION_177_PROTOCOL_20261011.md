# 177 TRAIN-only local-fit mechanism gate

The 176 frozen DEV result did not improve blind plane selection. Its two
global-correlation gradient steps also made the truth-best original branch
slightly worse on average. This one small mechanism assay asks whether a
fixed-support, spatially local atlas match with bounded line search can move
an already-near blind candidate toward the actual plane. It is **not** a
model training run, blind selection result, or permission to benchmark.

Use the frozen 132 direct model and its 16 blind proposals without passing
truth to proposal generation or to fitting. Draw 128 independent TRAIN
sections, 64 each from v3/v4, with fresh random physical planes, deformations,
artifacts, and backgrounds; record complete provenance. Diagnostic truth may
select the closest matching-reflection proposal only **after** all proposals
have been generated; keep proposals initially 0.5–3.0 mm from the true
five-point plane. If fewer than 30 eligible cases remain in either generator,
the assay is inconclusive. Do not choose steps using truth or inspect DEV.

Compare the unchanged 176 two-step 48-pixel global absolute-Pearson fitter to
a 96-pixel fitter with 30% global and 70% 9×9 local absolute normalized
cross-correlation. For each original proposal, hold its eroded atlas support
fixed; use three normalized tangent steps, candidate step lengths 0.1, 0.25,
and 0.5, and select by fit energy with a trust bound, support-retention term,
and update-size penalty. Apply exactly the same optimizer and support to a
within-support intensity-permuted atlas null. A plain no-step candidate must
remain available. The null is a causal control: background and brain support
must not explain a claimed anatomy-specific gain.

Report eligibility, initial and corrected physical five-point errors, fraction
improved, fit score, support retention, update size, angular strata, and
full-minus-null differences separately for v3/v4; include source, input, and
output hashes. Do not use oracle branch selection to claim blind performance.

Predeclared continuation gate: in **both** generators, require at least 0.15 mm
mean reduction from the *same original branch*, at least 65% of eligible
branches physically improved, at least 0.10 mm full-over-permuted-null
advantage in mean corrected physical error, and mean fixed-mask support
retention at least 99%. If any gate fails, do not enlarge this optimizer or
train another scalar evidence ranker unchanged. Pivot to better global
proposals and spatially anchored learned correction before another DEV look.
