# Joint adaptation/rehearsal003: valid experiment, local gate failed

Training terminal38572 exited0 after all4,000 applied FP32 updates,2153.03s.
Source at launch was8b4d58f. The independent post-exit audit
`training/audit_joint_v6_local_refinement.py` exited0 (terminal84611), with
`integrity_valid=true`, global retention passing, and `expansion_eligible=false`.
No output artifacts were accessed before the training process exited.
Run: `I:/AnatomyTracker/runs/joint_v6_joint_rehearsal_003`.
Audit: `I:/AnatomyTracker/runs/joint_v6_joint_rehearsal_audit_003/audit.json`.

This tests whole-model source/shared-feature adaptation plus broad proposal
replay together, relative to coordinate002's frozen-source local learning.
It uses the same whole original003 step20,000 parent, not the failed002 endpoint.
The exact first32,000 generated and32,000 frozen presentations of original003
are replayed; no new query observations are added. Local starts are truth-near
and PSF is known. Organizational groups from one atlas are not biological animals.

## Local endpoint

All256 fixed development rows are retained; the original eligibility masks
govern geometry. Metrics are eligible organizational-group macros.

| Readout | Paired geometric start | Coordinate002 endpoint | Rehearsal003 endpoint |
| --- | ---: | ---: | ---: |
| Five frame landmarks,um | 1325.5066 | 1239.5542 | 1238.2967 |
| Antipodal plane normal,degrees | 6.618312 | 6.609558 | 6.608613 |
| Pullback endpoint,px | .2725096 | .2762186 | .2772173 |
| Joint dense CCF correspondence,um | 2308.2619 | 764.1636 | 788.7624 |
| Reflection correct | identity tie baseline | 98.7463% | 97.4665% |

Landmarks improve6.5794%, below the required20%; pullback error worsens1.7276%
instead of improving10%. Normal error changes negligibly. Dense CCF error
improves65.8287%, but the local gate requires all components, not merely success
at resolving reflection. Unfreezing/replay gains only1.26um in landmark error
over coordinate002 and worsens its map/CCF/reflection readouts. The intervention
does not rescue conditional geometric learning under this schedule.

Every brush mode has worse pullback error than its identity baseline; accurate
brush also has worse normal error. These regressions independently preclude
expansion. There are zero nonpositive Jacobians over682,335 valid-tissue pixels;
their minimum is0.91920872 (whole-canvas minimum0.89879053). This is measured
before discrete reflection, not inferred from a mean of per-row minima.

## Global retention endpoint

These are the611 support-eligible rows of the fixed640-row global development
set, comparing this run's own FP32 step0 with step4000, not historical AMP.

| Readout | Step0 | Step4000 |
| --- | ---: | ---: |
| Full-cell NLL,nat | 8.097193 | 7.815908 |
| MAP plane-normal error,degrees | 43.528581 | 42.334536 |
| Exact-cell top128 recall | .270224 | .320317 |
| Antipodal frame angle,degrees | 72.659460 | 69.603809 |
| Normal-offset error,um | 2058.9556 | 2000.9030 |

All predeclared overall and mode-specific retention bounds pass. This is
preservation/modest improvement of a weak global model, not adequate accuracy.
Imperfect-brush offset error increases1787.00→1884.64um, within its allowed10%
tolerance; passing retention does not mean every metric improves.

The audit independently recomputes geometry, topology, ranks and normalized
full-catalogue probabilities. Raw finite values, provenance, source/receipt
bindings, exact replay schedules, train/development identity separation,
unchanged frozen tensors and all4,000 optimizer counters pass. The largest
full-catalogue log-normalization error is5.73e-7. It is a valid negative
learning result, not a training crash or integrity failure.

## Decision

Do not promote or extend rehearsal003. Preserve it for evidence and continue
with the predeclared [fresh image-key retrieval experiment](IMAGEKEY_RETRIEVAL_PROTOCOL_20260929.md):
learn slice-to-rendered-atlas matching with a shared image descriptor and full
gallery endpoint evaluation. It starts a fresh whole random model and compares
against original003 at4,000 updates, not this24,000-update lineage. No independent
encoders or old weights are merged. Separately develop the native curved-slab
joint path with coherent-subject targets; its implementation is not yet training
evidence. No calibration, external benchmark, DeepSlice superiority or shipping
claim follows from this result.

Exact SHA256 receipts:

- Audit: `c77e3c2cdedbe4b2ad1c81a7c8a85909bdf1446db4889a92b616f8b010804483`.
- Training completion: `cf83119f544f48ca21a7c46f8b898409c467b98ad9547d950350910b8613d63d`.
- Experiment: `385fd2a037504bf5cc249e137700b28c4f85281e829748a6d4aa65e3be285036`.

The audit includes the full frozen artifact hash inventory, raw recomputed
per-row metrics and every mode/reflection/censor stratum and gate decision.
