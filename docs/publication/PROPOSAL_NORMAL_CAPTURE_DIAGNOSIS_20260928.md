# Coarse proposal failure: normal capture, not merely full-cell recall

CPU-only diagnosis of **completed frozen 001/002 step-10,000 predictions**.
Neither recovery004 nor curriculum003 outputs were accessed. Results below are
the original saved probabilities, not corrected-head re-inference; 002 is
confounded by the separately documented mixed-precision defect.

The 98,304 cells are ordered as 384 normals × 16 offsets × 16 rolls. Exact
log-sum-exp marginalization gives
`joint NLL = normal NLL + offset|normal NLL + roll|normal,offset NLL`.

| All 640 development rows | Uniform reference | Frozen 001 | Frozen 002 |
| --- | ---: | ---: | ---: |
| Normal NLL | 5.9506 | 5.685893 | 6.741141 |
| Offset given normal NLL | 2.7726 | 1.980596 | 2.561171 |
| Roll given normal and offset NLL | 2.7726 | 1.913818 | 2.568876 |
| Marginal-normal MAP error, degrees | — | 50.475148 | 49.826863 |
| Joint-cell MAP normal error, degrees | — | 50.829943 | 49.772090 |

Approximately 86% of 001's joint-NLL improvement over uniform comes from the
conditional offset/roll terms. Normal localization barely improves; switching
from joint-cell MAP to marginal-normal MAP does not fix it. Full-cell metrics
include normal, offset and roll and are not the sole success criterion.

Among the 611 point-supervised rows, 001 marginal-normal error is 50.583°.
Section-mean errors by input mode are raw 55.000°, accurate brush 47.067° and
imperfect brush 49.596°. Excluding low-support rows does not explain the failure.
These mode summaries are descriptive section means, not biological-animal
estimates. All-row means equal synthetic-group macros because all 40 groups
contain 16 sections; groups are not distinct biological anatomies.

## Input/target and data-volume evidence

The 5,120 training rows cover 5,024 unique catalogue targets; each normal has
only 4–23 examples and just 4.375% of development exact-cell targets occurred in
training. Geometry interpolation/generalization is doing almost all the work.
The target cell's normal differs from exact truth by only 2.876° on average and
5.654° at maximum; the nearest normal alone averages 2.835°. Joint cost chooses
a non-nearest normal in 8.906% of rows, but discretization cannot explain 50°.

Seventeen fixed rows spanning pose/joint, all input modes and both horizontal
reflection states were inspected directly, without replaying or rehashing the
generator: indices `0,43,85,128,170,213,256,298,341,383,384,426,469,511,554,596,639`.
Prepared image arrays exactly match their frozen row arrays. Their valid dense
CCF coordinates lie on the saved physical target plane within 0.001655 µm.
This checks plane association/axes and input ordering, not every image-formation
operation or dense inverse-map convention.

Source tracing is consistent: finite pose/joint paths use the effective rendered
parent geometry; joint G1 is affine-free section deformation; image, dense
coordinates and pullbacks are reflected together. Canonical pose stays unchanged
under horizontal presentation reflection, which has its own raster affine.
Only horizontal reflection is sampled, not an unlabelled vertical/180° roll
augmentation. No concrete input/label-ordering bug was found.

## Head limits and next decision

The image encoder has only three convolutions (5/3/3 kernels, strides 1/2/2):
an 11-pixel local receptive field before spatial pooling. Its pooled spatial
features are compressed to a 64-dimensional context in 002. Each of eight
components scores learned 16-dimensional geometric bases. This is a plausible
capacity/inductive limitation, not a demonstrated cause of the observed error.

The density is **not three completely independent marginals**: `d*n` couples
normal and offset, and `(v,u outer n)` couples normal and roll. At fixed normal,
each component does factorize offset × roll; the conditional mixture has rank
at most eight. A sharp single mode only needs rank one, so that rank bound alone
does not explain failed basic capture. There is no explicit image-to-rendered-
atlas matching in this proposal-only training phase.

Proceed with the prepared fresh mixed curriculum003, corrected FP32 head,
160,000 generated observations and the frozen complex-row mixture. Follow
normal-marginal NLL and physical capture, not only total NLL. If normal capture
still stalls, the smallest controlled intervention is the same fresh-seed/data
schedule with `joint_NLL + normal_marginal_NLL`, initially coefficient one,
before changing architecture substantially. Both terms remain proper scoring
objectives for their respective distributions, but finite-capacity calibration
must still be validated; this does not confer calibrated uncertainty.

If clean generated rows learn but complex development rows do not, prioritize
appearance/shape diversity and consistent subject variation. If even clean
normal capture fails, test a direct normal readout and then deeper multiscale
image encoding or rendered-atlas contrastive retrieval. A first generated
training batch can diagnose fit, never unseen anatomy or generalization.

`training/audit_joint_v6_normal_capture.py` is prepared for CPU post-run use after
003 exits. It audits initial/final raw posteriors, retains per-section normal
probabilities and errors, reports synthetic-group macros by support/input mode,
and adds 10° posterior mass and explicitly oracle top-eight-normal capture.
It does not run model inference or modify frozen input artifacts. It has not
been executed against 003.

## Frozen provenance

Both runs share baseline `internal_development_prepared.pt` SHA-256
`89e078582aaa365299d9823deb2a005308f360ab8909db7125a4534556d622c4`
and `catalogue.pt`
`9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71`.
Data manifests and generator receipt are recorded in
`arbitrary_plane_v6_substantive_data_recovery.md`.

| Run | Artifact | SHA-256 |
| --- | --- | --- |
| `joint_v6_proposal_substantive_001` | `experiment.json` | `90bde7ec17104e340f03aa71913d581a5f9373d50862f0699d4acfba1453be10` |
| Same | `development_log_probability_step_10000.npy` | `70c9dfc9a55617341244d60c603935f283f5dced84d689fcc996e7950765cd1b` |
| `joint_v6_proposal_capacity_002` | `experiment.json` | `0a35059a0b314016e077471c485132ee2a993166256dd05b71ac259655a2e55a` |
| Same | `development_log_probability_step_10000.npy` | `06017cd080ece3563db4898f3a0725af958a42061391aea275aeb942fc58c96d` |

Paths are relative to `I:\AnatomyTracker\runs`. Raw probability hashes were
computed directly once for this note. The corresponding checkpoint hashes and
matched training schedule are recorded in the existing baseline/capacity audits.
