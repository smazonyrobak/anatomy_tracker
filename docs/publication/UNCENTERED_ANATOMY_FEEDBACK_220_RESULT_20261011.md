# 220: absolute similarity improves the matcher, but not blind registration

The TRAIN-only 220 ablation changed one scoring choice in 219: it stopped subtracting each atlas candidate's own mean source–atlas similarity before aggregation. The 187 step-800 pose model stayed frozen. Both arms used the same seed, 300 independently drawn 166→217→v4 TRAIN planes, eight fixed TRAIN-inner cases, parent checkpoint, optimizer, and 160-branch blind readout. All 300 training rows matched on draw seed, base, frozen prior error, and truth-chosen best-branch error; all eight inner cases matched likewise. The completed receipt's eight output hashes, six current source hashes, and parent checkpoint hash were independently checked. No reserved DEV/CAL/FINAL material, physical-oblique test, public benchmark, or calibration was used.

Errors are mean CCF displacement over surviving tissue sites, then the median of eight inner cases; they are **not** directly comparable to the 187 affine five-frame-point result. These eight cases have limited, mostly low-exposure tissue and cannot establish full-sphere generalization.

| Matcher, update | Frozen prior top-1 | Anatomy-only top-1 | Anatomy + 0.3 prior top-1 | After fitted correction | Truth-chosen best of 160 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 219 centered, 300 | 3.006 mm | 3.996 mm | 3.601 mm | 3.770 mm | 1.755 mm |
| 220 uncentered, 0 | 3.006 mm | 4.118 mm | 3.601 mm | 3.701 mm | 1.755 mm |
| 220 uncentered, 100 | 3.006 mm | 4.004 mm | 3.284 mm | 3.338 mm | 1.755 mm |
| 220 uncentered, 300 | 3.006 mm | 3.405 mm | 3.006 mm | 3.020 mm | 1.755 mm |

At update 300 the 220 fused selector chose the frozen prior branch on seven of eight cases, and chose a *worse* branch on the eighth. Its median antipodal normal error remained 60.0° and its <2-mm top-10 recall remained 4/8. The fitted correction moved the median slightly in the wrong direction. On the 300 training draws, removing centering improved exact-target versus best-branch score ordering from 21 to 86 cases and exact-target versus sampled-wrong ordering from 47 to 102. Nevertheless, the best available branch worsened after fitting in 178/300 draws (versus 219/300 in 219), with mean error 1.733→1.852 mm; the exact target gauge drifted 0.729 mm on average. This is a real mechanistic improvement in the score, not an accuracy gate pass.

**Decision:** do not promote 220, extend this matcher unchanged, claim calibrated uncertainty, or install it in the GUI. The frozen pose proposals themselves are weak on the 166→217 full-sphere distribution (truth-chosen best 1.755 mm / 27.2°), and the branch remains unable to turn anatomical fitting difficulty into reliable blind selection or safe deformation. The 187 step-800 parent had only 2,400 legacy synthetic presentations and no 217 support-aware framing; this distribution shift is a plausible contributor, not proven to be the only cause. The next bounded experiment should improve proposal training with matched old-domain rehearsal and a stratified TRAIN-inner/real-donor retention gate before revisiting fit feedback.

Frozen code is commit `f559878`; frozen run is `I:/AnatomyTracker/runs/blind_anatomy_feedback_220_uncentered_train_001`. The run's `completed.json` pins the config, three checkpoints, three raw inner-case files, and 300-row training log.
