# 131: matched-metric pose capture on independent v3/v4 draws

The failed 130 training logs reported a low rate of near candidates under a
five-point-plus-normal threshold, while the 128 DEV panel reported high beam
capture using valid-tissue mapping error. Those are different metrics and
different plane distributions; the numbers must not be compared directly.

Freeze the 128 treatment step-6,000 direct pose model and its exact blind-16
beam rule. Draw one eligible 256-pixel physical section from each of the 64
independent TRAIN local-deformation bases under v3 appearance and one
*independently drawn* section from each base under v4. The seeds, virtual
variants, plane, appearance and damage draws are independent across cohorts;
there are no deliberate same-plane appearance pairs. Preserve retries,
source/checkpoint hashes and full per-draw provenance. This is a TRAIN-domain
mechanism diagnosis, **not** held-out biological or synthetic-DEV accuracy.

For both cohorts compute the frozen 128 evaluator's mean 3-D rigid-plane
discrepancy over observed valid tissue pixels. Report best-of-beam and direct
prior top-one error and within-1.5-mm capture; stratify descriptively by
background/brush mode, nearest-cardinal angle, applied exposure and observed
tissue intensity. This rigid-gauge metric evaluates the pre-artifact fitted
plane at observed pixels and is not a displaced-fragment mapping metric.
Avoid causal language because the two cohorts are independently sampled, not
paired, and 64 sections per cohort give noisy strata. If v4 looks materially
worse, check the same metric on a larger, fresh independent draw before
changing direct-pose training. If similar, the strict five-point threshold
and training/DEV plane distributions remain candidates for the discrepancy.
Do not train, calibrate, promote to GUI or use a public benchmark in this
diagnostic.
