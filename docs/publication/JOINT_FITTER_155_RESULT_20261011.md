# 155 atlas fitter: useful near correction, exact-plane drift still needs damping

The scratch-lineage 155 continuation trained the 154 checkpoint's atlas matcher
and local tissue map for 6,000 new arbitrary-plane synthetic TRAIN sections.
The pose predictor was frozen. Exact, 0.35–1.5-mm nearby, 1.5–3-mm wrong,
reflection-mismatched, and top-16 on-policy alternatives appeared in the
two-candidate training pairs. These were new independently sampled planes, not
recolored copies. All 6,000 accepted presentations and 9,720 attempts came
from 60 TRAIN deformation plans; 48 held-out TRAIN sections from four other
plans selected checkpoints. None is a separate biological animal.

The selected update was 4,000. On those 48 held-out TRAIN sections, nearby
planes improved from 0.899 to 0.578 mm mean rigid error, and the exact-plane
fit energy was lower than a wrong plane in 45/48 cases. But a supplied exact
plane shifted 0.267 mm on average, above the predeclared 0.250-mm no-harm
limit. The 6,000-update checkpoint was slightly worse on this selection score.
**The prespecified feedback gate failed; Stage C joint training did not run.**
The run and all source/output/checkpoint hashes passed the independent 155
verifier.

A separate, frozen 64-case v4 synthetic DEV assay used eight deformation plans
disjoint from TRAIN, selecting eight cases per plan by section-ID hash rather
than model error. It compared the same exact/near/wrong candidates with the
154 parent, selected 155, and selected 155 with atlas intensity set to zero
while support remained. A correction gain of 0.8 was *assayed*, not trained:
it linearly scales the rigid correction from the supplied plane.

| Arm on matched v4 DEV cases | Exact drift, full correction | Near gain, full correction | Exact drift, 0.8 correction | Near gain, 0.8 correction | Exact fit energy below wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| 154 parent, atlas intensity | 0.402 mm | 0.264 mm | 0.321 mm | 0.272 mm | 53/64 |
| 155 selected, atlas intensity | 0.288 mm | 0.317 mm | **0.230 mm** | **0.307 mm** | **59/64** |
| 155 selected, intensity zeroed | 0.568 mm | 0.055 mm | 0.454 mm | 0.082 mm | 52/64 |

This supports the targeted next change: keep the 155 weights, cap the affine
correction at 80% of its present value, then test the *actual recomputed
fitted tissue map* and fit-to-pose learning path. The 0.8 numbers here are
posthoc rigid interpolation, not a newly trained 156 checkpoint or a final
mapping result. The DEV panel has been reused in prior research decisions;
fresh plans are needed for independent confirmation. Zeroing atlas intensity
is an out-of-distribution ablation for the full-trained matcher, not a matched
support-only training arm, so its collapse suggests reliance on atlas
intensity but does not establish a clean causal advantage over a properly
trained support-only model.

The illustrated [fixed-case grid](I:/AnatomyTracker/reports/joint_pose_154_fresh_v4_examples/true_input_guessed.png)
shows canonical atlas truth, actual artifacted generator input, and the 154
blind guessed plane for five prespecified appearance/support strata. It
illustrates both recoverable anatomical features and stark failures; it is
not selected by model performance. It also shows why a tiny fragment can
leave the plane ambiguous despite full-angle coverage in the generator.

Frozen outputs: `I:/AnatomyTracker/runs/joint_fitter_155_train151_001`
(selected checkpoint SHA-256
`aa2bbfc2c3894fbaad3ce732a011632911b83842a7a4acb58959f344ce282179`)
and `I:/AnatomyTracker/runs/joint_fitter_155_damping_dev_assay_001`.
Both independent read-only verifiers passed. No physical oblique
animal, calibrated uncertainty, GUI replacement, final test, or DeepSlice
benchmark participated.
