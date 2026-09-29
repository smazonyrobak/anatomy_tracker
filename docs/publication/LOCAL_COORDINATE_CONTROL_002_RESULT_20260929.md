# Local coordinate control002: completed, learning gate failed

Run `I:/AnatomyTracker/runs/joint_v6_local_coordinate_control_002` exited0
(terminal70095) after all4,000 applied FP32 updates, 1,760.69 seconds.
Source at launch was `866ee1e`. The post-exit independent NumPy audit,
`training/audit_joint_v6_local_refinement.py`, also exited0. Its output is
`I:/AnatomyTracker/runs/joint_v6_local_coordinate_audit_002/audit.json`.
No run artifact was accessed while training was active.

The matched intervention adds only896 zero-initialized coordinate-evidence
weights. The whole003 parent, all local schedules, data, starts, losses and
optimization settings match001. Source/global proposal features remain frozen.
This is conditional truth-near local learning with known synthetic PSF, not
image-selected global registration. All256 development rows are retained;
231 have identifiable geometric targets. Group macros weight organizational
synthetic groups equally; these are not independent biological subjects.

| Group-macro readout | Paired geometric start | Local001 endpoint | Coordinate002 endpoint |
| --- | ---: | ---: | ---: |
| Five physical frame landmarks, um | 1325.5066 | 1245.0737 | 1239.5542 |
| Antipodal plane normal, degrees | 6.618312 | 6.609075 | 6.609558 |
| Pullback endpoint, px | 0.2725096 | 0.2836333 | 0.2762186 |
| Joint dense CCF correspondence, um | 2308.2619 | 810.9474 | 764.1636 |
| Reflection correct | identity-prior tie baseline | 96.1868% | 98.7463% |

Landmarks improve6.4845%, below the20% requirement. Pullback error is1.3611%
**worse** than identity, rather than the required10% reduction. Normal error
barely changes. CCF error falls66.8944%, but much of this resolves discrete
reflection; unreflected rows improve only820.3739→752.5717um. Reflected rows
improve3925.5299→778.2189um. No promotion or budget extension is justified.
Coordinate002 improves the001 landmark endpoint by just5.52um; the coordinate
hook alone does not solve the poor conditional geometry learning.

| Brush mode | Eligible rows | Landmarks initial → final, um | Normal initial → final, degrees | Map identity → final, px |
| --- | ---: | ---: | ---: | ---: |
| Absent/raw | 79 | 1306.7086 → 1225.2458 | 6.75986 → 6.74552 | .2686823 → .2713351 |
| Accurate/black | 79 | 1376.0693 → 1297.5633 | 6.86102 → 6.89898 | .2581411 → .2644943 |
| Imperfect | 73 | 1294.1368 → 1196.9977 | 6.25746 → 6.18925 | .2885079 → .2901153 |

Map error worsens in every brush-mode macro. Accurate-brush normal error also
regresses. Reflection-stratified and mode×reflection results, including these
regressions, remain in the raw audit rather than being removed from reporting.
There are zero nonpositive Jacobians over682,335 original valid-tissue pixels;
the actual minimum is0.88600085 (whole-canvas minimum0.88160718). The topology
measurement precedes the discrete reflection. A group mean of minima is not
the global minimum.

The audit independently reconstructs OUV, landmarks, normals, map/CCF errors
and Jacobians. Maximum row disagreement with the runner is0.00266um for
landmarks,0.00100um for CCF and7.36e-6 degrees for normals. Parent initialization,
frozen tensors, exact schedules, IDs, hashes and all4,000 optimizer steps pass;
all saved predictions/parameters are finite. All19 original trainable tensors
change and the new coordinate weight changes by L2=0.47725284. This is a valid
negative experiment, not an execution/integrity failure.

## Decision and next experiment

Retain the whole003 parent. Select the already prepared
[joint adaptation plus broad proposal rehearsal](JOINT_GLOBAL_REHEARSAL_PROTOCOL_20260929.md)
as the next fixed4,000-step experiment. It starts from that same parent with
coordinate conditioning, unfreezes source/shared features and adds full-cell
proposal replay. It does not merge independently trained encoders or continue
the failed002 endpoint. Evaluate local geometry and global retention under its
predeclared gates. Failure of the frozen-feature control motivates that new
intervention; it is not retrospectively called a pass. No public benchmarking,
uncertainty calibration or delivery claim follows from these experiments.

## Exact receipts

- Audit JSON: `7f14f9bb46de2cd6ce0bcd3572ea53b4137365d9375acd6bbc7b52ef338dbb01`.
- Final checkpoint: `58f2b893883197f05cfe15760d9088682bd98c3116b273fe4e4379a549b8f6a2`.
- Final raw predictions: `e52df676f00004ce57e0c386c16726345e7ed355d3a019b627946e4a00bc6eaf`.
- Fixed schedule: `1efe598bf513a8adf63735914790e8e5262c6255d834a9716cb216180a5a193f`.
- Whole parent: `a011063fc44d2f9c331fe63c743797492ab40f4a72e1e8f72d5dd1b035c297a2`.

The audit contains exact hashes of config, driver snapshot, trace, completion,
packs and both endpoint artifacts, plus its own source hash.
