# 131: dark synthetic sections lose frozen pose capture

The frozen 128 treatment model was evaluated without updates on 64 eligible v3
and 64 eligible v4 synthetic TRAIN sections. Each cohort used one independent
physical plane from each of the 64 local-deformation subjects; the cohorts did
not reuse planes as paired appearance variants. There were 149 draws to obtain
128 eligible sections, and all 128 physical-section and plane hashes were
distinct. The exact blind-16 beam and observed-valid-pixel rigid-gauge error
from the 128 evaluator were used in both cohorts. This is a mechanism diagnosis
on TRAIN synthetic subjects, not held-out biological validation.

| Appearance | Sections | Best beam mean | Best beam within 1.5 mm | Direct top-one mean | Direct top-one within 1.5 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 64 | 0.968 mm | 56/64 (87.5%) | 2.351 mm | 33/64 (51.6%) |
| v4 | 64 | 2.004 mm | 26/64 (40.6%) | 3.857 mm | 12/64 (18.8%) |

V4 applies independently sampled exposure to the same *generator family*;
the two cohorts' actual planes, artifacts, backgrounds and virtual-affine
variants were independently drawn. The loss of capture appears within the
15–30°, 30–45° and 45–55° nearest-cardinal-angle strata, though the groups
are small and angle distributions differ. V4 exposure shows a strong
descriptive gradient: best-beam near capture was 3/25 at exposure <0.15,
6/16 at 0.15–0.30, 5/10 at 0.30–0.60, and 12/13 at 0.60–1.20. Mean valid
tissue intensity was 0.480 for v3 and 0.162 for v4. These observations are
consistent with a brightness-sensitive direct pose model; independent draws
do not prove exposure is the sole cause. The high-exposure v4 subgroup is
small and its anatomical/angle distribution is not matched to low exposure.

The 130 training log's 13.9% five-point-plus-normal near-candidate fraction
must not be compared as an identical metric with these 40.6% or 87.5%
observed-valid-pixel rigid-gauge rates. The current output also does not
measure final fitted tissue mapping or displaced-fragment recovery.

**Decision:** verify the large v3/v4 capture difference on a larger fresh
independent draw from the same 64 TRAIN maps, with exposure and cutting-angle
strata, before adapting the direct pose encoder. If confirmed, address the
appearance/domain gap and restore pose capture before another fit-score head.
Do not promote 130, calibrate probabilities, integrate the GUI or benchmark
DeepSlice on this evidence.

Frozen output: `I:/AnatomyTracker/runs/v3_v4_pose_capture_131`. Completion
receipt SHA-256: `302b5ec5f6202bf4d6298f0e701f04a0a0db0167fdea0e834e9e13f1d3d8beda`.
All four output hashes and the evaluator-source, parent-checkpoint and parent-
config bindings matched. The first two attempts exited before creating the
output because old and new TRAIN-map lineages have different plan-ID fields;
the completed script uses their common `subject_id`, which was checked unique
across all 64 bases. The two corrected source versions are retained in Git.
