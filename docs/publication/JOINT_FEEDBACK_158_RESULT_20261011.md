# 158: retained-optimizer joint feedback did not establish a useful advantage

The matched comparison completed normally and its independent read-only
verifier passed source/checkpoint/receipt hashes, all 4,000 accepted TRAIN
physical-section draws per arm, exact cross-arm draw pairing, the 48 fixed
held-out TRAIN sections per checkpoint, and checkpoint-selection arithmetic.
Both arms began with the same randomly initialized training lineage: the
selected 154 pose weights and AdamW moments plus the frozen selected 155 atlas
matcher/tissue-map weights and 156 damped rigid correction. Each update
accumulated gradients from four fresh independent arbitrary-plane sections.
The only arm difference was whether a supervised post-fit rigid error could
backpropagate through the fitted atlas comparison into the live pose state;
the control detached that state. No external model weights, real images,
public benchmark, or final animals entered this run.

The predeclared inner-selection score was mean direct first-choice error +
0.5×mean physically best-of-160 error + 1−top-16 near-candidate fraction.
Lower is better. Both arms selected update 500:

| Held-out TRAIN synthetic readout | Shared update 0 | Feedback 500 | Detached control 500 |
| --- | ---: | ---: | ---: |
| Direct first-choice rigid error | 4.170 mm | 3.700 mm | 3.766 mm |
| Physically best of all 160 (oracle diagnostic) | 1.257 mm | 1.225 mm | 1.200 mm |
| A ≤1.5-mm candidate in direct top 16 | 32/48 | 30/48 | 31/48 |
| Near-candidate post-fit rigid error | 0.942 mm | 0.810 mm | 0.803 mm |
| Selection score | 5.132 | 4.688 | 4.720 |

The 0.066-mm first-choice difference favoring feedback is too small and
inconsistent to credit to the fitting loop. Feedback was better on 22/48
paired sections, its median paired difference was actually +0.008 mm, and the
plan-level first-choice differences (feedback minus control) were −0.280,
−0.032, +0.014 and +0.033 mm across the four synthetic inner plans. Its
best-of-160 and top-16 coverage were slightly worse than control. At update
1,000, feedback first choice was 3.807 mm versus control 3.861 mm, again a
small difference; neither was selected. The common improvement over update 0
is consistent with continuing the direct pose optimizer with its retained
moments and four-section updates; this paired study does not isolate which of
those two protocol changes mattered.

**Decision:** do not promote 158 as evidence that fitting feedback improves
global pose prediction, and do not scale this loss unchanged. The next compact
diagnostic freezes the selected arms and tests whether the existing untuned
atlas-fit score can select a correct plane from each model's actual blind top
16, rather than from an injected correct/near plane. A separate donor-equal
weak-real guard tests acquired-image transfer; its Allen affines are inherited
weak references, not expert truth. Both scripts were committed before their
runs. Their results, not the encouraging absolute inner-score decrease, will
direct the next architecture/data change.

The source audit in [JOINT_GENERATOR_158_SOURCE_AUDIT_20261011.md](JOINT_GENERATOR_158_SOURCE_AUDIT_20261011.md)
also found that this run used only 60 fixed local deformation maps and omitted
the available virtual-affine, support-point and acquisition-framing branches.
That is a concrete data-distribution limitation; it is not demonstrated to be
the cause of the feedback result. All sections and inner plans descend from
one Allen atlas, not independent biological animals. There is still no
calibrated location/trajectory probability, GUI replacement, physical
all-angle validation, or justified DeepSlice comparison.

Frozen run: `I:/AnatomyTracker/runs/joint_feedback_158_train151_001`.
Selected checkpoint SHA-256 values: feedback
`02144c37a4a146bd08d97429090d54aa93c7ff82b061c81e5fdd7cd439651d7c`,
detached control
`5b8e145b324e65e8a6649e94b64c75b99c5f5cd750d1565ccb445510af4d5e74`.
