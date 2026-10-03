# 047 joint pose-fitting feedback: gradient present, localization did not improve

The first training batch confirmed a finite, nonzero fitting-evidence-only gradient from the atlas matcher, differentiable plane fit, and warped atlas score to the pose head's coordinate outputs (norm 911). The 500-batch pilot finished normally in 487.7 s, with 500 newly drawn eligible synthetic TRAIN sections from 64 synthetic animal/deformation identities; 546 independent draws were recorded, including 46 ineligible sections. Peak GPU allocation was 1.49 GB. The frozen 019 mapper, 041 coarse matcher, and 046 fine matcher remained differentiable but were not updated. No pretrained external weights, public benchmark, final-test animals, or calibrated probabilities were used.

The predeclared comparison on all 177 eligible sections of the unchanged eight-identity synthetic 037 DEV panel finished in 167.1 s. Errors are mean observed-tissue-point distance to known synthetic CCF truth, equally averaged over the eight synthetic identities:

| 047 training batch | Posterior-selected mapped error | Best-of-four mapped error (oracle diagnostic) | Posterior-selected rigid-plane error |
| ---: | ---: | ---: | ---: |
| 0 | 2.440 mm | 1.126 mm | 2.453 mm |
| 250 | 2.556 mm | 1.127 mm | 2.567 mm |
| 500 | 2.551 mm | 1.117 mm | 2.565 mm |

The required ≥15% improvement and ≤1.5-mm selected error were both missed. At batch 500, selected mapped error by appearance was raw 2.398 mm, exact-black 2.661 mm, imperfect-brush 2.502 mm, versus initial 2.334, 2.580, and 2.317 mm. All three appearances worsened; none worsened by >10%. This checkpoint is **not promoted**.

The failure is mostly **choosing the wrong branch**, not losing the good branch. At batch 500, the selected-to-best-four gap was 1.469 mm, up from 1.343 mm initially, while best-four geometry barely changed. The fitted scorer was worse than simply choosing prior top one by 17, 76, and 106 µm on average at batches 0, 250, and 500; it helped only ~24–25% of sections. The residual of the eight-query coherent match correlated weakly with actual selected CCF error (Pearson r≈0.17 at batch 500). A small residual therefore does not certify that anatomy was placed in the right atlas region. A differentiable feedback path alone does not supply a reliable difficulty signal.

The next targeted change is to learn *candidate selection* from independent, spatially distributed evidence and explicit synthetic physical-error supervision, including support and warp cost. Before further training, test whether those candidate-level observables distinguish the physically good branch from wrong-but-internally-coherent branches. Do not spend more batches on the 047 pose-only feedback objective or claim it outperforms 019. A fresh identity-disjoint synthetic panel and expert-labeled biological animals are still needed before any deployment/calibration decision.

Frozen artifacts: `I:/AnatomyTracker/runs/joint_pose_fit_047_pilot` and `I:/AnatomyTracker/runs/joint_pose_fit_047_development_eval`. A read-only audit confirmed 500 accepted presentations, 546 unique physical draw IDs, disjoint 64 TRAIN and eight DEV animal IDs, 177 finite DEV rows, and matching frozen draw/row SHA-256 hashes. Training draw hash `1e57a2b814c1ad877a32bf6edf0ea6d5d10bd0a128b154d56b868f4d62e279aa`; evaluation row hash `e19641f620e226c66f42d15fa0b456315de25386b7e94b4b3e06949bb083cc07`.
