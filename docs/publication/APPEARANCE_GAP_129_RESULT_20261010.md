# 129: acquired-image exposure gap and a provisional generator correction

The earlier v3 arbitrary-plane generator did not match the brightness scale of
the acquired sagittal-like images used in training. A read-only comparison used
160 acquired TRAIN sections from 12 donors, 80 weak-reference DEV sections from
four separate donors, and 40 independently generated, eligible raw synthetic
sections whose normal lay within 20° of the ML axis. Each real number below is
the median of donor medians; each synthetic number is the section median.
All images were measured at 256 × 256 on the same nominal [0, 1] input scale.

| Per-image statistic | Acquired TRAIN | Acquired weak DEV | v3 synthetic | v4 synthetic |
| --- | ---: | ---: | ---: | ---: |
| Mean intensity | 0.0317 | 0.0289 | 0.1892 | 0.0353 |
| 95th-percentile intensity | 0.0927 | 0.1066 | 0.7716 | 0.1672 |
| Intensity standard deviation | 0.0304 | 0.0399 | 0.2464 | 0.0586 |
| Mean absolute XY gradient | 0.00680 | 0.01143 | 0.03512 | 0.00758 |
| Fraction of pixels ≤ 0.004 | 0.6369 | 0.6040 | 0.6327 | 0.7203 |

The v3 synthetic sections were roughly six times too bright by image mean and
eight times too bright at the 95th percentile relative to these acquired
TRAIN sections. This is a concrete domain gap, not evidence that localization
would improve by a corresponding factor. The v3 and v4 synthetic panels used
different independently drawn planes, not duplicate training examples with
different artifacts; the sample is small and this is a descriptive check, not
a matched causal effect estimate.

`arbitrary_plane_one_shot_slide_artifacts_v4.py` now draws an independent
exposure per section: 55% low exposure in [0.07, 0.27] and 45% higher exposure
in [0.30, 1.20], log-uniform within each range. Low-exposure tissue receives
0.002–0.008 read-noise standard deviation, high-exposure tissue 0–0.003. The
existing v3 physical-plane, geometric distortion, missing tissue, background
and optional-brush sampling remains independently random. Noise is restricted
to visible tissue, preserving exact-black exteriors where requested. The
high-exposure branch remains because the other acquired modalities are not
uniformly as dark as this sagittal-like cohort. No synthetic section is made
twice as a systematic black/non-black or artifact pair during training.

The v4 summary is closer in mean and image-gradient scale, but its bright tail
and fraction of near-black pixels remain broader than the acquired sagittal
cohort. The four-row actual-generator preview contains independently sampled
coronal, sagittal, horizontal and steep-oblique planes with their fitted atlas
planes and pre-/post-artifact images. It is a generator quality-control figure,
**not** model predictions. Visual inspection still shows smooth, stylized
anatomy compared with granular physical histology. This exposure correction
alone does not validate appearance realism, arbitrary-plane accuracy or the
failed 128 pose-selection gate. Do not scale 128 unchanged on v4 or promote a
checkpoint on this basis.

The frozen 128 rows also expose a scoring defect: its original action retains
the direct-prior score, while only the corrected action receives the learned
match logit. A post-exit, read-only diagnostic selected the original action by
corrected score instead. On treatment, plan-equal original-pose error changed
from 3.3002 to 3.2682 mm; within-1.5-mm capture remained 26.3%. Removing the
visual features while retaining candidate geometry/support gave 3.2144 mm,
better than using the learned visual features. Thus fixing score symmetry is
necessary for a fair feedback path but, by itself, is not the needed anatomical
discriminator. This posthoc selection was not a trained or deployed model.

Source and outputs: `I:/AnatomyTracker/data/allen_sagittal_ish_weak_inputs_001_20261008`,
`I:/AnatomyTracker/runs/appearance_gap_129/v3.json`,
`I:/AnatomyTracker/runs/appearance_gap_129/v4.json`, and
`I:/AnatomyTracker/runs/appearance_gap_129/branch_score_128.json`, and
`I:/AnatomyTracker/previews/actual_v4_independent_planes_20261010.png` with its
JSON receipt. SHA-256 of the two audit JSON files: `4d325be0af7be5cd779f11ff88bbd33cbf318451ba5f7b60a0a1dfbccdd1c36c`
and `35d6bbc74557c43710a71536dbd7e359d2f5578a81f78790bead6ab574397305`;
posthoc score JSON: `1eff17e7f94914dd41a7c3b8df1f69fff58b85f514a9ed503f99d191495f45c0`;
preview PNG: `4a76cd0fc1de8d08a98c72274b18defc88093c0d686f34741743a5e705382605`.
