# 167: current atlas-fit loss has a weak local pose direction

The bounded [TRAIN-only diagnostic](../../training/diagnose_joint_fit_gradient_direction_167_train.py) froze the scratch-lineage 132 step-2,000 model, drew 128 new independent brain-intersecting 166+v4 sections from 60 TRAIN deformation bases, and selected **one blind top-prior pose per image**. It differentiated the exact 162 measured fit objective—local image/atlas LNCC mismatch, atlas coverage, warp cost and rigid-correction cost—through the finite-thickness atlas renderer and fitter. The synthetic tissue-validity mask weighted this *training diagnostic*; it is not an inference-time input. True pose was used only to score results and construct an explicit oracle-direction control. No weights were updated.

At each predicted pose, one fixed local step (Euclidean length 0.5 in coordinates scaled to 5° rotation, 500 µm translation and 0.05 local basis/shear) followed the measured fit gradient, a seeded random direction, or the true rigid-error gradient. After the learned atlas-matcher correction, mean rigid-error changes across all 128 nonblank sections were:

| Direction | Mean change (negative is better) | Fraction improved |
| --- | ---: | ---: |
| Measured fit descent | −0.0179 mm | 57.8% |
| Random unit direction | +0.0025 mm | 54.7% |
| Oracle rigid-error descent | −0.1555 mm | 100% |

The measured-versus-random paired advantage was 0.0204 mm; 37/60 TRAIN-base means were positive, with a descriptive base-resampled 95% interval of +0.0013 to +0.0382 mm. It is not an animal-level confidence interval. The measured/oracle gradient cosine averaged 0.211 overall, 0.305 on 58 retrospectively near (≤1.5 mm) poses, and 0.133 on 70 farther poses. Near and far measured steps improved rigid error by only 0.0280 and 0.0095 mm respectively. The fit objective itself decreased after the fixed step in 68% of cases. LNCC supplied the largest gradient component in 125/128 cases; coverage dominated three, while warp and correction gradients were much smaller. All 128 gradients were finite and nonzero.

**Decision:** the current loss contains some local pose information, but its small, heterogeneous descent and weak far-pose direction cannot plausibly resolve the millimetre-scale first-choice/global-selection failure, consistent with the failed 162 paired training gate. Do not scale this scalar feedback unchanged or promote a checkpoint. Next, train an anatomy-sensitive, supervised fit-quality/correction signal on *blind predicted candidates*, then allow that fitted evidence to teach direct pose ranking/coordinates in the same model; retain a matched detached control and source/atlas-dependence checks. The 166 support-point stream is useful for efficient intersecting draws, but sparse tangential sections still need an explicit stratum. This diagnostic did not test real animals, calibration, GUI use or a public benchmark.

Frozen output: `I:/AnatomyTracker/runs/joint_fit_gradient_direction_167_train_001`. Its completion receipt matches all four output and current source hashes. Raw per-case scores, full synthetic lineage/plane/appearance provenance, config and draw schedule are preserved there. No DEV, final-test or public benchmark was accessed.
