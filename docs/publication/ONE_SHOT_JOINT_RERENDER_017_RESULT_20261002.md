# Joint atlas-rerender continuation 017 — held-out development result

Frozen training: `I:/AnatomyTracker/runs/one_shot_joint_rerender_017` (16,000 batches, two fresh arbitrary-plane synthetic examples and one distinct reserved real training section per batch). Parent: own-lineage 015 batch 8,000. The atlas-fit evidence entered a shared pose updater; a second differentiable atlas render and fitted-map score sent gradient to the global pose head. The batch-one fit-evidence-to-pose gradient was nonzero (norm 4013). No legacy or external learned weights were used.

Frozen evaluation: `I:/AnatomyTracker/runs/one_shot_joint_rerender_017_development_eval`, 185 eligible synthetic cases across eight synthetic deformation plans and 64 real sections across six separate development donors. The real targets remain inherited Allen affine alignments, not blinded anatomical truth. `python -B -m training.verify_one_shot_joint_rerender_017` passed: 16,000 training rows, 32,000 accepted synthetic examples, 16,000 distinct real training sections, all nine checkpoint hashes and 2,241 raw development rows. No public benchmark or calibration was used.

Identity-equal synthetic means (mm):

| Checkpoint | Direct mapped | Refined selected mapped | Best-eight rigid oracle | Correct-pose mapped | Real weak-label refined five-point |
| --- | ---: | ---: | ---: | ---: | ---: |
| 017 batch 0 | 2.799 | 2.784 | 1.168 | 0.129 | 0.552 |
| 017 batch 14,000 | **2.727** | 2.789 | 1.198 | 0.126 | **0.542** |
| 017 batch 16,000 | 2.885 | 2.790 | 1.202 | 0.128 | 0.561 |
| 015 parent batch 8,000 | 2.799 | 2.706 dense-head selection | — | 0.129 | 0.619 direct; 0.688 dense |

The predefined gate required at least 0.25 mm improvement over the 015 direct 2.799 mm and no more than 0.2 mm real weak-label regression, while retaining sub-0.2 mm correct-pose mapping. **It failed the synthetic accuracy requirement.** The best trained refined-selection result (2.789 mm at batch 14,000) improves direct parent by only 0.010 mm; it does not even beat the 017 batch-zero refined selection (2.784 mm). Direct prediction itself reaches 2.727 mm at batch 14,000, a small 0.072 mm gain, but refinement then worsens it. The 015 dense synthetic result of 2.706 mm at its batch 8,000 (2.693 mm at batch 12,000) remains lower, although that dense head failed real weak-label retention and cannot be considered qualified.

The fitter still maps an exact-pose input to roughly 0.13 mm, so local mapping capacity is not the immediate limit. The best of the eight proposed branches is about 1.2 mm away, but the fitted selector returns about 2.8 mm. At batch 16,000, refinement changes the best-eight rigid oracle by only 3.4 μm in absolute value on average (maximum 19.6 μm); it is not meaningfully correcting pose. The selector changes branches in 84/185 cases, helping 92 and hurting 93 relative to direct mapping. At batch 14,000 the lowest-tissue quartile (visible fraction ≤0.097; 47 cases) has 4.07 mm case-mean refined error versus 3.95 mm direct, with 15/47 above 5 mm; the other 138 cases average 2.37 mm refined. The six real weak-label donor means range from 0.225 to 0.760 mm refined; these are not independent expert-registration errors. This is a selection/global-proposal failure, not evidence that a further recurrent step alone will fix it.

Next: perform one targeted frozen-panel diagnostic of per-branch score versus physical error and atlas-fit evidence, then change the ranking/proposal learning mechanism based on that evidence. Do not call these development figures real-animal accuracy, calibrated uncertainty, or DeepSlice superiority. GUI support for loading this whole-checkpoint architecture exists but is not yet validated or qualified.
