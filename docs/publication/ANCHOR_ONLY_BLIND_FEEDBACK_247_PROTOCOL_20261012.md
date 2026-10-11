# 247: one randomly initialized anchor-only pose/warp model

This is a bounded corrective training experiment, not a shippable model. The
[244 result](SHARED_ANCHOR_JOINT_FEEDBACK_244_RESULT_20261012.md) and
[245 candidate diagnosis](SHARED_ANCHOR_CANDIDATE_SCORE_245_RESULT_20261012.md)
show that a free-frame branch monopolized the blind output, the normal-anchor
bank was rarely considered, fitted ranking worsened physical error, and the
local warp was effectively zero. Do not continue the 244 checkpoint or train
another scorer on frozen 244 outputs. Initialize **all weights from a new
seed** and keep code, data, outputs and temporary files on I:.

Use the same encoder, 64 antipodal normal anchors with shared image-conditioned
pose residuals, finite-thickness atlas renderer, atlas-conditioned local map,
shared pose updater and fitted-candidate scorer as a **single model**, but
remove the 16 competing free-frame modes. Every predicted plane can now be
assigned to a distinct orientation anchor; normal mode collapse remains
possible and must be measured, not assumed away. A plane has continuous
centre, in-plane rotation/scale, bounded tangent correction and reflection.
This is a full-sphere representation, not a cardinal-family restriction.

Sample two independently randomized v7 physical sections per batch from the
64 TRAIN-only synthetic deformation bases, with independent planes, roll,
warp, artifacts and backgrounds. Also sample one distinct acquired Allen
TRAIN section per batch, donor-uniformly, as a low-weight appearance/weak-
pose guard. The acquired affine is not anatomical truth, and no real weak
label supervises fine deformation. Retain all donor, specimen, experiment,
section, synthetic-physical-section, seed and source hashes. Run an initial
6,000 batches (12,000 unique synthetic draws; 6,000 distinct real TRAIN
sections), saving 0/1,000/3,000/6,000. This is a pilot exposure, not an
arbitrary one-million-presentation target.

Teach the normal distribution explicitly from each synthetic plane's
antipodal normal using a fixed 15° soft anchor target. Train the complete
plane (centre, frame, scale and reflection) against surviving-tissue physical
sites for the three nearest anchors, weighted by their normal-target mass;
also minimize expected site error under the 64-mode/2-reflection posterior.
This prevents a normal-only classifier from certifying a wrong location. Use
the acquired weak affines only as low-weight five-point references. Train the
exact-pose local map against 512 sampled surviving observed-pixel-to-CCF
targets per synthetic section, record its no-warp baseline, and regularize
local displacement and roughness. Weight this physical map error fivefold
relative to its millimetre-scale pose term, since the 244 map loss was much
smaller and its field stayed near zero. This is a fixed training choice, not
selected after viewing 247 results.

After batch 1,000, form eight *blind anchor* proposals per feedback example:
the two reflections of each of the four highest-mass **distinct** normal
anchors. Render through the recorded PSF, update pose once, re-render, map
and score. Train the combined fitted score to rank physical error **within
those eight blind candidates**, with expected physical task loss; do not mix
an injected exact/near-true branch into that softmax. Separately perturb the
true plane to teach near-pose correction and mapping. Do not add the pose
refiner's auxiliary score logit to fitted score unless that logit is trained.
This uses the differentiable
hypothesis-scoring principle of [DSAC](https://openaccess.thecvf.com/content_cvpr_2017/html/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.html),
adapted to histological atlas sections; [SynthMorph](https://pmc.ncbi.nlm.nih.gov/articles/PMC8891043/)
supports contrast/deformation randomization, but neither paper proves this
specific architecture will work. The score may depend on atlas intensity,
candidate geometry and local deformation cost; a shifted-intensity atlas is
the prespecified negative control at readout.

Before training, freeze the same-code development readout. Read checkpoints
0/1,000/3,000/6,000 on the **same 32 synthetic and 12 weak-real sections**
used in 244. Select the checkpoint with the lowest matched-32 fitted-selected
rigid error; report all four without hiding a worse late checkpoint. Then
read *that one checkpoint* on every eligible independent synthetic DEV
section and every acquired weak DEV donor section, conserving computation
while testing angular coverage. Report identity-equal selected/best-of-eight rigid
and mapped physical errors, normal-axis error by angle, top-one anchor
diversity, exact-pose map versus zero warp, raw/black/brush and visibility
strata, and intact-versus-null ranking. The null shifts atlas intensity while
preserving support, then reruns the same fixed prior beam through refinement,
mapping and fitted ranking; distinguish any fixed-state rescoring from this
end-to-end control. Best-of-eight is truth-informed
diagnostic only. Do not use synthetic DEV sections or acquired DEV donors in
training, and do not open CAL/FINAL or public DeepSlice data.

Advance to substantially larger exposure only if the selected checkpoint's
matched-32 fitted-selected rigid error falls at least 0.40 mm below 244
batch 6,000's 4.037 mm, intact atlas improves matched-32 selection over its
intensity-null by at least 0.25 mm on average and in at least six of eight
synthetic plans, the donor-equal weak-real refined disagreement is at most
3.5 mm, and the exact-pose map beats the matched zero-warp error by at least
10%. On the full 185-section all-angle panel require at least four selected
anchor IDs each used on at least five sections, and the 75th percentile
antipodal angular spread of selected physical normals from their projective
medoid to exceed 25°. On its 21 eligible
45–55° sections require mean selected rigid error at most 3.5 mm and median
selected normal-axis error at most 20°. These remain coarse development
thresholds, not an acceptable final error. A new single-branch or
physical-normal collapse fails regardless of mean error. Passing is only a
development continuation gate,
not a physical heavy-oblique, GUI, calibrated-uncertainty or public-benchmark
qualification. If it fails, diagnose the failed term; do not buy batches
on the same objective by default.
