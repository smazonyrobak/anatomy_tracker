# 132: short mixed-exposure direct-pose adaptation

The frozen 131b diagnostic established that low-exposure v4 sections lose
blind pose proposals even before fit scoring. Test a targeted continuation of
the project's own randomly initialized 128 treatment step-6,000 model. Keep
the network architecture and frozen atlas/local-map/fitted-score weights.
Train the shared image encoder/lateral features and 16 base plus 64 anchor
probabilistic pose heads at small learning rates. Each batch draws **new,
independent** physical TRAIN sections: two v4 and one bright v3, never the
same section with a second background, plus one acquired coronal and one
acquired sagittal TRAIN image with their documented weak Allen affines.
Retain complete source/section/donor provenance and each rejected draw.

Use the 128 direct physical-pose mixture loss, a v3/real retention term and
brief 0/500/1,000/2,000-batch checkpoints. This is a capture intervention,
not a claim of completed joint fitting: freeze the atlas matcher and local
mapper to isolate the proposed pose mechanism. Evaluate each checkpoint and
the frozen parent on identical, independently prepared v4 sections from the
eight existing synthetic DEV deformation plans, the existing v3 DEV panel,
and the donor-disjoint acquired coronal/sagittal weak-DEV sections. Report
best-blind16 and top-one observed-valid-tissue rigid error/capture at 1.5 mm,
angle/exposure/mode/plan strata, and weak-real five-point disagreement per
donor. These DEV plans and weak labels have informed this project; they are
not untouched biological animals, true physical oblique labels or a public
benchmark.

Select a checkpoint for further *development only* if it improves v4
blind-16 capture by at least 15 points and v4 direct top-one capture by at
least ten points over its same-panel parent, loses no more than five points
of v3 blind-16 capture, and increases no real weak-DEV donor's mean five-point
disagreement by more than 0.2 mm. The first two thresholds address the
mechanism; retention matters because the CNN is shared with mapping. If no
checkpoint passes, do not extend this training unchanged. If one passes,
next verify correct-pose and blind-selected local-map accuracy with the
changed encoder before any recurrent fitting/GUI promotion. No calibration,
animal-final testing or DeepSlice benchmark is opened by this experiment.
