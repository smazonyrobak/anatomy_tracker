# 176: blind-branch atlas-fit correction and selection

The frozen 174 learned corrector did not use atlas intensity. The 175
TRAIN-only gate found a real polarity-invariant finite-thickness image-fit
direction *only inside a near, correct-reflection basin*. This stage tests
whether that signal can become an inference rule over the unchanged blind
16-branch beam, without oracle choice. It keeps the 132 image encoder,
probabilistic 80-mode plane proposals, reflection logits, and tissue mapper
frozen; a small randomly initialized fit-evidence ranker is the only learned
stage-1 component. No prior external weights, features or pseudolabels enter.

Each candidate receives two short, bounded full-frame updates guided by the
autograd derivative of 48-pixel, 3-pixel-smoothed, polarity-invariant Pearson
agreement between the observed image and a finite-thickness atlas render.
The scoring mask is fixed from the candidate's own initial eroded atlas
support, independent of synthetic truth or a supplied brush. The update must
be first-order and bounded; the original state is always a no-shift action.
The ranker sees the blind prior and explicit fit evidence (before/after
correlation, improvement, gradient magnitude and support retention), not
synthetic coordinates. A within-support intensity permutation is a matched
anatomy-mismatch control. Synthetic truth can supervise which of the 32
original/corrected actions is physically best during TRAIN, but cannot enter
the beam, rendering mask, update, fit features, or inference action score.

Train on fresh independently sampled v3/v4 TRAIN physical sections from all
64 procedural deformation bases, one of each appearance per optimizer update.
No deliberate reuse of a physical plane with a different background. Save
draw-level provenance and the fixed 64-base inner TRAIN panel. Start with
500 updates, preserving the ranker and optimizer checkpoint; extend the same
lineage to 1,000 and 2,000 updates only if evidence warrants. Separate run
directories record each continuation and its parent checkpoint hash.
The frozen development comparison uses the existing 248 v4 and 243 v3
eligible synthetic sections across eight plans per version, plus separate
donor-equal weak coronal/sagittal readouts. Evaluate original selected error,
blind corrected/no-shift selected error, 1.5-mm capture, truth-chosen
best-of-16 availability (oracle diagnostic only), and the same scores after
atlas intensity is permuted within tissue support. Keep all raw rows and
hash receipts. Do not use final animals or public DeepSlice benchmark.

A meaningful stage-1 advance should improve v4 blind selected error by at
least 0.30 mm and 1.5-mm capture by at least 10 percentage points, show a
material anatomy-dependent advantage over the permuted control, and avoid a
serious v3 or acquired-donor regression. These are decision aids, not evidence
of calibration or deployment. If blind ranking fails despite good near-branch
correction, revise retrieval/ranking rather than scaling a local fitter from
the wrong branch.

Only after blind branch selection works should the same model jointly fit the
existing affine-projected, derivative-bounded tissue map, then compare
matched full versus detached atlas-fit feedback into the direct pose head.
Keep an explicit physical warp-effort penalty so local deformation cannot
erase a wrong global-plane cost. At that point, verify a finite *useful*
atlas-fit gradient reaches the direct pose heads and improves independent
development errors; a nonzero norm alone is insufficient.

The design follows the iterative render/transform-update mechanism of
[SVoRT](https://pmc.ncbi.nlm.nih.gov/articles/PMC10129054/) and the
image-similarity plus smoothness principle of
[VoxelMorph](https://arxiv.org/abs/1809.05231), but does not presume either
establishes success on arbitrary-plane histology. The eventual external
[DeepSlice](https://www.nature.com/articles/s41467-023-41645-4) comparison
must account for its coronal-only setting and separate real-image labels.
