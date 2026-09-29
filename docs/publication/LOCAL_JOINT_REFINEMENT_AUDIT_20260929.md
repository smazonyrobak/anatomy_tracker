# Local joint refinement 001: completed, continuation gate failed

Session `6220` returned exit code zero before the output tree was accessed.
All 4,000 optimizer updates applied in 1,647.84 seconds. The independent CPU
audit `training/audit_joint_v6_local_refinement.py` (preserved at commit `4c3a6f0`)
then returned exit zero.
Audit success authenticates this **failed learning result**, not model quality.

This is the predefined truth-near, known-PSF, one-atlas local experiment, not
image-selected arbitrary-plane capture or biological-animal validation. Of 256
development rows, 231 have eligible pose and dense targets. Metrics below average
within organizational synthetic groups, then across groups. They are not evidence
of performance across real animals. No external benchmark or calibration was run.

| Physical metric | Paired starting error | Final error | Decision |
| --- | ---: | ---: | --- |
| Five canonical-frame landmarks | 1325.5066 um | 1245.0737 um | 6.07% reduction; fails 20% criterion |
| Antipodal plane normal | 6.61831 degrees | 6.60907 degrees | Negligible improvement |
| Pullback endpoint | 0.272510 px | 0.283633 px | 4.08% worse; fails 10% reduction criterion |
| Dense CCF correspondence | 2308.2619 um | 810.9474 um | 64.87% reduction, largely reflection resolution |

The starting correspondence baseline uses the actual perturbed frame, identity
map and fixed identity-reflection tie-break, not true reflection. Final reflection
accuracy is 96.19%. For truly unreflected rows, CCF error **worsens** from 820.37
to 853.80 um; for reflected rows it falls from 3925.53 to 765.41 um. Thus the large
aggregate CCF reduction does not establish successful native pose/deformation
learning. The common deterministic frame/warp still warrants further development.

| Brush mode | Eligible rows | Landmark initial -> final (um) | Normal initial -> final (degrees) | Map initial -> final (px) |
| --- | ---: | ---: | ---: | ---: |
| Absent | 79 | 1306.71 -> 1223.47 | 6.75986 -> 6.73185 | 0.268682 -> 0.278791 |
| Accurate | 79 | 1376.07 -> 1302.79 | 6.86102 -> 6.91867 | 0.258141 -> 0.272226 |
| Imperfect | 73 | 1294.14 -> 1211.89 | 6.25746 -> 6.18779 | 0.288508 -> 0.297566 |

Map error worsens in every brush mode. Accurate-brush unreflected rows are a
particularly poor subset: reflection accuracy 83.33%, CCF 787.37 -> 1023.85 um.
Do not conceal this with pooled reflection gains. The audit retains all brush x
true-reflection combinations, exact per-row recomputations and group aggregates.

## Integrity and topology

Independent NumPy state-to-OUV decoding, antipodal normals, five landmarks and
observed-pullback -> predicted reflection -> CCF recomposition agree with the
saved metrics: maximum row differences are 0.00266 um for landmarks, 0.00094 um
for CCF, 0.00000736 degrees for normals, and 0.000000085 px for map error.

No nonpositive pullback Jacobian occurs, including the 682,335 valid-tissue pixels;
the minimum is 0.8424343. This is checked **before** discrete reflection, which is
not a deformation fold. Safe topology alone does not imply accurate registration.

Initial model keys and every tensor equal the complete own-lineage 003 parent.
All final frozen tensors remain identical; all intended trainable module groups
changed. Parameters, raw arrays, loss/gradient traces are finite; optimizer states
record 4,000 updates. Exact schedules, development perturbations, pack receipts,
record identities and non-overlap of all five recorded group/section IDs pass.
These IDs remain organizational one-atlas groups, not independent brains.

## Decision and next experiment

The original numerical continuation gate fails. Do not extend local001 as if
native geometric learning were demonstrated, calibrate its uncertainty, run a
public benchmark or claim a deliverable. Proceed to the predeclared matched
[coordinate-evidence control 002](LOCAL_COORDINATE_CONTROL_002.md), retaining the
same whole 003 parent and every other experimental setting. This tests a specific
spatial/readout hypothesis, not an established implementation bug.

If frozen-source controls remain inadequate, source-feature adaptation with
global proposal rehearsal must be considered as a new experiment. The engineering
gate is not a reason to permanently freeze features that may need registration
training. Such a change would be reported separately, not retroactively counted
as local001 passing. Subpixel deformation-loss scaling is another unproven
hypothesis and is deliberately not changed in the coordinate comparison.

## Frozen evidence

- Run: `I:/AnatomyTracker/runs/joint_v6_local_refinement_001`.
- Audit and independent row arrays: `I:/AnatomyTracker/runs/joint_v6_local_refinement_audit_001`.
- `audit.json` SHA-256: `f9e2deb948be28cbf9bf803a07ea45a9896644e32f5c4565573402f4a7bc0d9c`.
- Final checkpoint SHA-256: `969ce6b74a651a3139f9c9a14e571dc71d2d1cc6efdee432318e741e9f0ad34e`.
- Final raw predictions SHA-256: `995c4c7c480fa9ff45e77969a9bbe2f882fbf36db5df7e484716758106c654bc`.
- Schedule SHA-256: `1efe598bf513a8adf63735914790e8e5262c6255d834a9716cb216180a5a193f`.

The audit also records hashes of the full parent, data, executed source,
experiment, completion record, trace and both zero/final endpoints. No frozen
run file was modified by the audit.
