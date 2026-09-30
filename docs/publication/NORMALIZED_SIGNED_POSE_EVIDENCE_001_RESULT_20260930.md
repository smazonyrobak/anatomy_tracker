# Fixed TRAIN-RMS signed-pose control: completed, not promoted

Runner90795 exited0 from `2a86507`, completing2,000 updates/8,000 training
presentations. The independent CPU audit completed after runner exit. Its former
terminal handle3256 is now absent and no Python process remains; the completed
audit, not the missing handle alone, establishes the result.

Integrity passes; the unchanged scientific gate **fails**. The only failed
geometry condition is `oracle_normal_overall_20pct`. All coordinate20% gates
and populated mode/subject nonregression conditions pass. No threshold waiver,
extra updates or scalar-normalization sweep is justified.

## Main result

Eligible synthetic-subject macro values,270/384 development observations from
four held synthetic subjects; the114 censored observations remain in raw results.
These are not four independent biological animals. Starts are truth-near,
PSF is supplied, and "oracle" means the recorded correct reflection is selected.

| Readout | Unscaled signed2k | Fixed-RMS signed2k |
| --- | ---: | ---: |
| Correct-reflection centre error, um | 709.93149 | 644.06732 |
| Correct-reflection slab error, um | 709.93426 | 644.07016 |
| Correct-reflection plane-normal error, degrees | 6.10189 | 5.87902 |
| Automatically selected centre error, um | 842.27370 | 960.62547 |

Thus correct-reflection centre improves65.86417um but selected centre worsens
118.35177um. Normal error improves only0.22287degrees against the unscaled
control. Against the geometric initializer (~814.24um,6.13834degrees), this is
about20.90% coordinate improvement but only4.22% normal improvement, below20%.
The geometric initializer is distinct from the fresh network's step0 output;
the independent audit reconstructs and reports both, and gates against the former.
Reflection selection also worsens. The normalization improves one part of the
conditional problem, but has not resolved plane learning or branch selection.

## Independent evidence and exact bindings

All paths below are under `I:/AnatomyTracker/runs/`.

- `joint_v6_normalized_signed_pose_evidence_001/completed.json`:
  `57554cc00bcd4446409cb3cd3bc0dd0fb11320565e32b4f1b97a4c7bc33331c1`.
- Its final whole checkpoint `joint_model_step_02000.pt`:
  `023a2bc3a5d8fbb8c57f5e00957c6380c960c6b00caaaf812a4f6c65effb165f`.
- `joint_v6_normalized_signed_pose_evidence_001_independent_audit/audit.json`:
  `c42402dbfb5d74c790aff62e9bbc59b4cbebb60f79618474016e1a931dbe16d8`.
- Exact shared schedule:
  `e8aaa6eb1ff7e70097cbf1eeb1f6b9fc4d2ba4f01f26fc6d91b671c90c8dc0b9`.

The audit authenticates source/config/schedules/raw predictions/checkpoints and
unchanged retrieval tensors; reconstructs final state/residual/director geometry
and spatial-only reflection with independent NumPyFP64; recomputes metrics and
gate decisions against both the geometric initializer and unscaled endpoint.
Maximum mean-metric difference is0.001876um, coordinate recomposition0.005608um,
and normal difference0.000005762degrees. Weighted-p95 disagreement up to3.986um
is retained diagnostically because of discrete FP32 cumulative-weight ties.
Topology evidence authenticates the reported all-branch/all-iteration checks;
it is **not an independent topology replay**. Encoders/renders/gradients were
not independently replayed. Runner elapsed3153.17s is not a hardware benchmark.

## Decision

Proceed with the predeclared TRAIN-only feature-Jacobian versus learned-readout
diagnostic on exact prior TRAIN observations. It uses the whole **unscaled**
signed model, oracle fixed fields and single-axis starts, so remains an off-policy
diagnosis, not a qualification or comparison against the latest normalized head.
Separately address coherent-image field-of-view transfer in coarse training.
No benchmark, calibrated uncertainty, GUI readiness or biological superiority
claim follows from this result; never splice these independently trained native
heads into the separately continued coarse model.
