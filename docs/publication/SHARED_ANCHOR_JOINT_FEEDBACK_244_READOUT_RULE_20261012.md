# 244 development-checkpoint rule (frozen before the 244 readout)

Use only the four saved steps 0, 1000, 3000, and 6000 and the fixed
synthetic/weak-real development panel in the 244 evaluator. No calibration,
final animals, or public benchmark enter this choice.

For each step, use the evaluator's animal-identity-equal mean errors. Define
the dimensionless development score relative to step 0 as

`0.45 * synthetic_fitted_selected_rigid / synthetic_step0_fitted_selected_rigid
 + 0.35 * synthetic_fitted_selected_mapped / synthetic_step0_fitted_selected_mapped
 + 0.20 * real_weak_refined / real_step0_weak_refined`.

The lowest score is the **diagnostic best checkpoint**. It can be considered
for further training only if all three gates hold: (1) intact-atlas synthetic
fitted-selected rigid error is lower than the atlas-null control error, (2)
real weak-label refined error has not risen more than 20% above step 0, and
(3) exact-pose synthetic mapped error is at most 1 mm. If no step passes,
declare the 244 objective unsuccessful; do not call the lowest-score step a
qualified model. These development metrics do not establish animal-level
physical arbitrary-plane accuracy, uncertainty calibration, or GUI readiness.

Training JSONL `synthetic_oracle_um`, `synthetic_prior_selected_um`, and
`real_weak_prior_selected_um` include an angular penalty. They are composite
training costs, not literal physical site errors. Use the evaluator's physical
site errors for all checkpoint decisions.
