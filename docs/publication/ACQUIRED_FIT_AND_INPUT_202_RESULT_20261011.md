# Acquired fitting and input-appearance checkpoint (202/195/200/203)

The TRAIN-only AL1A fit-ranking assay completed without optimizer updates. It used
the compatible 187 step-800 pose and its 193 adapted map, **not** the later 188
all-TRAIN pose. Among 52 source sections, 37 blind top-16 beams contained a
candidate within 1.5 mm of the approximate human plane and a candidate at least
3 mm away. On the same eroded common atlas-support pixels, the nearer plane
won 24/37 with no warp, 21/37 with the inherited warp, and 22/37 with the
adapted warp. The direct pose prior chose the nearer one in 26/37. Permuting
atlas intensity within that support gave 20/37 for the adapted warp. The
adapted map therefore did not make this fit score a useful pose-learning
signal; do not feed its current score back to the pose head. These are weak
human alignments from one TRAIN guard animal, not expert landmark truth or an
independent test. Frozen rows and hashes are in
`I:/AnatomyTracker/runs/lupori_fit_rank_202_train_001/completed.json`.

The 195 appearance audit sampled 32 distinct TRAIN fluorescence sections and
32 independent synthetic sections from each generator version. The old v4
generator's median tissue intensity was 0.159 versus 0.379 for acquired
sections after the same 0–1 preprocessing. The new v7 generator, which retains
the v6 physical-plane and warp draw and independently varies gamma, exposure,
and background, had median 0.364. Its median raw exterior is often brighter
than the acquired sample (v7 0.072; acquired 0.000), deliberately retaining
confusing-background coverage. This audit establishes an input-distribution
correction, **not** improved model accuracy. The 6-row true/input/blind-guess
grid is `I:/AnatomyTracker/previews/current_v7_true_input_guessed_194.png`.
Every row is a fresh physical-plane draw; it is not a paired background copy.

The Janelia MouseLight registered NRRD coordinate array needs physical
`(z,y,x)` mapped to Allen array `(AP,DV,ML)`. A five-plane visual/correlation
audit improved mean smoothed correlation from 0.687 under naive ordering to
0.781 under that permutation. The ML **sign remains unresolved**; broad
bilateral anatomy cannot determine laterality. Three registered CC-BY-4.0
volumes were resampled on I: only as exploratory appearance sources. The
2017-01-15 arbitrary-plane grid is sparse and frequently almost blank;
six blind 188 predictions ranged from 1.78 to 7.05 mm rigid error. These
volumes have **not** entered training and are not suitable site-level pose
labels without a side-labeled anatomical check. The registered data are
credited to [Janelia MouseLight](https://registry.opendata.aws/janelia-mouselight/);
the Allen array convention is described by the [Allen API](https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas)
and its [header clarification](https://community.brain-map.org/t/mouse-reference-atlas-nrrd-file-header-problem-and-question-on-axes/688).

The next model change must establish an anatomical fit signal that separates
correct from plausible wrong atlas planes *before* increasing fit-to-pose
gradient. Pose and deformation are still not jointly qualified; CAL/FINAL
animals, uncertainty calibration, GUI deployment, and DeepSlice comparison
remain untouched.
