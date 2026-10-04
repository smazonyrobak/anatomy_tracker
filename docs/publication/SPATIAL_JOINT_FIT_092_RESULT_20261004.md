# 092 result: spatial fitting still does not identify the right plane

The 092 pilot continued the project's own randomly initialized 089/083 lineage. It trained on 6,000 independently drawn, eligible arbitrary-plane synthetic TRAIN sections (6,651 attempts, 6,651 distinct physical-section IDs), one appearance per draw. Its 6,000-section checkpoint and independent frozen-beam evaluation both completed normally. No original Anatomy Tracker weights, external pretrained features, real DEV labels, final animals, public benchmark labels, or images were used in this experiment. Synthetic subject-plan IDs are not biological animals.

The architecture did connect tasks: a local 3-D atlas match was fitted to a full arbitrary plane; a fine atlas section was re-rendered; held-out match quality entered the candidate score and training loss; the pose, correspondence, and local map components trained together. Focused runtime preflight found a nonzero fit-energy gradient to pose. Across training audits, fit-only gradients reached the old and anchor pose heads 57 and 23 times, and ranking gradients reached direct prior logits 61 times. This establishes a learning path, not useful model behaviour.

On the exact 086 14-branch beam for 246 fixed synthetic DEV sections/eight deformation plans (plan-equal means), the prespecified advancement gate failed:

| Model/checkpoint | Score-selected mapped error | Best of 14 mapped error | Score versus lower error Spearman |
| --- | ---: | ---: | ---: |
| Inherited 089 final | 2.592 mm | 0.961 mm | — |
| 092 step 0 | 2.748 mm | 1.277 mm | 0.111 |
| 092 step 2,000 | 2.713 mm | 1.142 mm | 0.114 |
| 092 step 6,000 | **2.685 mm** | **1.130 mm** | **0.111** |

The 092 model improved during its own training, but the final result is 0.093 mm worse in selected error and 0.169 mm worse in best-of-14 error than 089. Its updated natural beam selected 2.682 mm, essentially the same failure. Final selected mapped error at 256 pixels was 2.685 mm, so the 96-pixel readout was not hiding a large resolution effect. Raw, exact-black, and imperfect-brush final selected errors were 2.822, 2.630, and 2.558 mm, all worse than 089's corresponding 2.630, 2.562, and 2.430 mm. The black-specific absolute gate passed, but the primary selected-error, score-association, and best-of-14 gates failed.

The failure is localized. Training pairwise rank accuracy rose from 65.8% in its first 1,000 draws to 70.7% in its last 1,000, and correct-match BCE fell from 1.104 to 0.840. Matched-state DEV correspondence CE did not regress: coarse 4.430→4.414, fine 4.052→4.037. Yet on blind DEV candidates, the fit energy alone selected 2.837 mm, worse than the direct prior's 2.611 mm; adding it to the prior selected 2.685 mm. The fit energy was almost entirely tied to how much atlas tissue was in the rendered search region (mean within-section fit-energy/atlas-support Spearman −0.961), while its association with actual mapped error was approximately zero. Reliability averaged over the plane was likewise not associated with candidate accuracy. The reliability target and matched-cell CE learned local training behaviour, but did not turn the aggregate fit score into an anatomical test for model-generated wrong planes.

The fitted transform also damages promising candidates: at step 6,000, the best available initial rigid branch averaged 0.997 mm; after spatial fitting the best rigid branch was 1.169 mm, and after mapping the best was 1.130 mm. By contrast, the exact synthetic local deformation relative to its known plane averaged only 0.155 mm on this panel. Thus the large current error is primarily wrong global placement/selection, not lack of local warp amplitude. An exploratory post-hoc algebraic removal of the empty-atlas contribution from the stored fit energy improved selected error only to 2.551 mm with the prior; that calculation is not the frozen 092 model, a new gate, or a qualification result.

Decision: do not promote 092, calibrate its normalized scores, integrate it into the GUI, or use untouched final animals or DeepSlice labels. Do not spend a very long continuation merely tuning this support-dominated scalar. The next change must make anatomically correct model-generated hard proposals distinguishable from lookalikes, using spatially ordered 2-D↔3-D evidence and a fit measure conditional on observed/atlas support; it must preserve or improve near-true candidate geometry. The earlier 050/090/091 scorer failures and 075/076 dense-field failures rule out simply adding another pooled score or repeating the same rank loss. The arbitrary-plane system remains experimental.

All 738 section-step groups and 11,531 candidate rows were independently re-counted; each group has exactly 14 fixed candidates and one score-selected and one physical-best branch. Recomputing the final plan-equal selected error from raw rows exactly reproduced 2,684.847667 µm. The training config/draw/log/checkpoint and evaluation config/candidate/section/summary SHA-256 values matched their completion receipts. Frozen files are under I:/AnatomyTracker/runs/spatial_joint_fit_092_pilot and I:/AnatomyTracker/runs/spatial_joint_fit_092_development_eval.

| Artifact | SHA-256 |
| --- | --- |
| TRAIN draws | 7676d81a60000d999f98304debe6b859d9e9c4b0d413e4582bc391da3fbfd0d2 |
| TRAIN log | 37356bbc1a7f99d29a7c4a3880cb595c673f686a5f683494e9d51f4c0d872c25 |
| Step-6,000 checkpoint | 4ed54ad8c19a242d81d6778edd79796bcab3a56f21e720b88f6985fbd11cb4d7 |
| DEV candidates | a9110015c95aa09b6b0ca51b2cb57e49a14480a573a5c55b5c84d784372deb78 |
| DEV sections | 86692829f5acfad049deaab6afc95a1d42a7461b72202714a39dab067576683d |
| DEV summary | 4052dc7cf7678605bb5c5bb0ffdb4a4bd498797a4dca3c8e8661aa29b22dd164 |

The original method lineage and relevant external precedent remain documented in the 092 protocol and prior studies. In particular, [SLIV-Reg's single-slice-in-volume feature matching](https://arxiv.org/abs/2410.18683) and [DeepSlice's coronal training/validation design](https://www.nature.com/articles/s41467-023-41645-4) inform the next design, but do not validate this arbitrary-plane histology result.
