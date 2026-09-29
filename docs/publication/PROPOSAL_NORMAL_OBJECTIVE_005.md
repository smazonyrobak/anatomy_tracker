# Normal-capture objective experiment — 2026-09-29

Curriculum003 completed 20,000 attempts (19,996 applied), with 160,000 rendered
observations covering all 98,304 catalogue cells. Its frozen final normal-
marginal NLL is 5.1841 versus uniform 5.9506, and marginal MAP normal error is
43.46 degrees. Only 11.41% of normal MAP estimates are within 10 degrees.
Additional coverage helped but did not establish a usable coarse initializer.

Experiment005 keeps the same fresh complete model initialization seed, model
widths, renderer, generated appearance/cell schedules, frozen-row order, batch
size, optimizer, learning rate and 20,000 attempts as003. It changes the objective
to `joint-cell NLL + normal-marginal NLL`, both weighted by the same point-pose
supervision weights. The normal marginal is the exact log-sum-exp over the
16 offsets and 16 rolls for each of the 384 catalogue normals; it is not an
independent head or a hard point-estimate loss. Coefficient one is fixed before
running. No weights from003, older projects or external pretrained models load.

The flat `training/run_joint_v6_proposal_curriculum.py` now targets
`I:\AnatomyTracker\runs\joint_v6_proposal_normal_objective_005`. The exact003
script is preserved in Git and in its frozen `experiment_source.py`. Updated
trace keys distinguish joint NLL, normal NLL and the summed objective. Evaluation
additionally reports normal-marginal NLL and marginal MAP angle. Organizational
ID prefixes change with the run namespace; this does not create new anatomies.

Compare matching steps and the fixed final endpoint against003, emphasizing
normal angle/capture as well as joint-cell recall and all brush modes. Do not
call the minimum development loss a final-test result. If normal capture remains
poor, investigate more expressive global image/pose features or explicit atlas
matching, rather than continuing the same objective indefinitely.

The opt-in recurrent covariance integration is disabled here. This remains
proposal-only training on one atlas plus frozen synthetic slices, not calibrated
joint registration, independent-animal validation, public benchmarking or release.
