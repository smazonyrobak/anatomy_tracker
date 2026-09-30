# Coherent canonical-plane adaptation: fixed TRAIN-only pilot

Root source review completed2026-09-30; enabled for a committed launch after
the short pose-readout diagnostic releases the GPU. No DEV/public-benchmark inputs or outputs are
opened. This is a training-effectiveness experiment, not a generalization,
registration, uncertainty or delivery qualification.

## Whole-model parent and fixed exposure

Continue audited whole A22576, checkpoint
`e22ff8e13a09b8c624c64518ca65a0ac9feda2c823dde30b6b8b5707995d52d8`,
matching gallery
`5219117c897bf68fb6139b5bd482bfdd05f5c52e2b9a5a7296c9936d0f8d0b36`,
independent audit
`c75c4ff817e7603eca31c8086080aafdb7f0953b3fec58c3f3f2f7842cf2afb3`.
Restore its exact model kwargs/tensors, AdamW state and RNG. Keep the same
histology stem, atlas stem, shared encoder and descriptor trainable; all other
parameters remain byte-identical. No module/encoder/refiner merging.

Run exactly 1,920 additional updates, batch24: eight coherent observations,
four old generated observations, four frozen synthetic rows and eight real
TRAIN images. Coherent seed2026092937 makes eight complete shuffled passes of
all1,920 observations:512 base plus128 acquired-view physical TRAIN sections,
all three original raw/exact-black/imperfect-brush presentations, the same
eight synthetic subjects. Retain every censored row; do not redraw it.
The first1,920 completed A schedules supply the old generated/frozen/real rows,
chart choices and candidate sampling. These are replayed observations, not new
unique catalogue coverage or biological animals. Existing real donor balance
and matched/canonical chart choices stay unchanged. AdamW LR0.00025,
weight-decay0.0001, clip5, temperature0.1, FP32 without AMP/TF32 are unchanged.

## New positive and separate loss

Use each **canonical anatomy fitted** O,U,V, not its already-observed fit.
Let c=O+(U+V)/2, u=U/|U|, v=normalize(V-u(u·V)), n=u×v, and support origin S.
The continuous canonical positive has centre c0=S+n[n·(c-S)], proper frame
[u,v,n], and orthogonal12mm basis. Thus it preserves plane and in-plane roll
while removing tangent-centre/span/shear nuisance. Do not assign nearest-cell
onehots or use the curved field as deformation supervision. The fitted plane
only summarizes genuinely curved anatomy; it is not exact dense correspondence.

Render640 canonical atlas intensity/support images once with the existing
normalized nine-point50um PSF; generate both representation images by spatial
horizontal flip. This fixed key PSF is not the observed25–100um PSF and is not
inferred from hidden target metadata. Original query pixels/FOV, optional user
outline and availability are unchanged. No canonical warp, resize to12mm,
truth support channel or target/ID/PSF input is supplied to the query encoder.
Current model features are recomputed for every training query/key. Cached
A22576 query/gallery descriptors are not training targets.

The new coherent loss is weighted paired sampled NCE on continuous canonical
anchors and the existing sampled catalogue negatives, with logmeanexp over two
representations and **no cell prior** for continuous keys. Per-query exclusion
uses antipodal normal≤10deg AND sign-aligned normal-offset≤500um; reinsert the
own positive. Same-plane/mode companions are not false negatives. Invalid
coherent anchors are excluded as negatives. Positive weight is original
finite/visible-support eligibility AND canonical-key support mass≥64; preserve
and explicitly report unsupported keys and censored observations, never redraw.
Support-overlap and plane-fit mismatch remain limitations, not new tuned gates.

Keep original synthetic sampled cell-NLL and real paired-NCE denominators and
exclusions unchanged: synthetic keys stay catalogue-only; real keys stay old
real anchors plus catalogue. New coherent anchors enter only the new loss.
Use exactly L=0.5 L_coherent+(1/3) L_synthetic+(1/6) L_real, preserving the old
2:1 synthetic:real ratio within the replay half. Each branch normalizes by its
eligible weight; an empty branch contributes zero, without weight redistribution.

## Fixed TRAIN endpoints and engineering decision

Evaluate only steps0/1920, all1,920 coherent+5,120 old synthetic+1,280 real TRAIN
rows. Build a fresh full98,304×2 gallery at each endpoint. Save current query
and gallery descriptors, original full-catalogue top128 cell/component masses,
conditional R probabilities, retained/omitted mass, row identities and geometry
readouts. Retained mass is not renormalized into an exhaustive beam posterior.
For coherent rows also save own/near-plane ranks against all640 continuous
canonical anchors, to distinguish paired-anchor learning from deployed-gallery
improvement. These ranks are diagnostic, not a replacement for full-gallery gates.

Coherent gates use the **original query eligibility**, including any rows whose
canonical key is unsupported (training-weight eligibility is reported separately):
eligible synthetic-subject-macro top1 plane angle improves≥5deg and physical
top128 plane capture improves≥0.10 absolute versus this run's FP32 step0.
No populated eligible corpus×mode capture128 may regress by more than0.02.
Old TRAIN retention: eligible overall and every populated synthetic mode, plus
all-row real donor macro, normal error≤baseline+2deg and capture32≥baseline−0.02.
All gates are arbitrary predeclared engineering screens, not confidence intervals
or biological validation. Save all rows and mode/group summaries, including
censored/ineligible rows. No checkpoint selection, extension, LR sweep or DEV
evaluation follows automatically if these gates fail. Passing only permits a
separate reviewed next step; it does not qualify the model.

Bind exact corpus/cache receipts, parent, schedules, pixels, source and outputs.
Preserve one whole model/optimizer/RNG lineage with periodic recovery checkpoints.
A changed encoder invalidates the old coherent candidate cache; subsequent native
training needs newly image-ranked candidates under that same whole checkpoint.
