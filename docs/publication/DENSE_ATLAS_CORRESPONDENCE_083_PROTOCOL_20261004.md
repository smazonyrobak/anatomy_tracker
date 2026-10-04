# Dense atlas correspondence 083: development protocol

## Question

The frozen 081 final checkpoint has a best-of-14 mapped error of 0.988 mm but
selects a 2.792 mm branch. The 082 geometric audit shows that a single
existing candidate contains 98.3% of exact visible coarse-grid matches,
whereas the score-selected candidate contains 81.2%. The 081 correlation
features were trained only through three-branch downstream pose/map losses;
the training ranking set was not the full 14-branch inference set. The next
test asks whether direct supervision of *actual synthetic tissue-to-atlas
matches* can give the shared image/atlas features a usable localization signal
before any further pose or warp fine-tuning.

[SAM/SAME](https://openreview.net/pdf?id=fo3erF5Uj0) show that dense anatomical
embeddings can guide registration in other modalities, and
[DISA](https://www.nature.com/articles/s41598-025-11583-w) learns a differentiable
histology-to-volume similarity. Neither demonstrates arbitrary-plane mouse
histology performance, so this is a mechanistic experiment, not a borrowed
accuracy claim. No external pretrained weights, features, or pseudo-labels
are permitted.

## Frozen first stage

Keep the internally scratch-trained 059 parent fixed. Initialize a new 083
whole-slice feedback head randomly with 081's physical cost-volume geometry,
but expose its coarse and fine local match logits. For each fresh independent
synthetic training slice, form the same blind 8-old/6-anchor beam. Train on
the physically closest *existing* candidate and one uniformly random other
candidate. At visible 16×16 and 32×32 query centres with the exact synthetic
observed-pixel-to-CCF target inside the search window, supervise the nearest
normal/lateral atlas cell with a local categorical match loss. Do not replace
a blind candidate with the true pose. Mask out-of-window points rather than
assigning false local targets, and require atlas support of at least 0.5 at
the quantized target cell; report how many visible points remain trainable.
Keep the full-angle generator and independently
randomized tissue plane, deformation, artifact, and background draw; never
make deliberate paired variants of one slice. Preserve every accepted and
rejected draw's provenance and source identity.

This first stage trains only the cross-modal match representation, not a new
GUI model. It has 6,000 synthetic presentations, checkpointed at 0, 1,000,
3,000, and 6,000 batches. Evaluate the same steps on frozen eligible 061
synthetic DEV plans with the fixed blind beam, recording coarse/fine masked
match cross-entropy, geometric in-range fraction, and physical top-1 match
error at query centres. Treat a held-out loss improvement, not training-loss
decrease alone, as evidence of learnable correspondence. Advance to the
second stage only if plan-equal coarse cross-entropy and top-1 physical match
error each improve by at least 20% from the frozen step-0 head, without fine
cross-entropy worsening by more than 5%. Report all three metrics at each
checkpoint; this mechanistic gate does not imply good pose accuracy. If
held-out matching does not clear it, stop before expensive joint fine-tuning
and inspect atlas
resolution, depth spacing, and appearance-domain mismatch. No public
DeepSlice benchmark or untouched final-test animals are used.

## Conditional second stage

Only if the first stage learns useful DEV matches, train pose, local mapping,
and quality jointly from that representation. Replace 081's fixed
three-branch rank training with existing-beam positive plus uniformly sampled
and hard-negative branches, so the quality objective covers the same branch
distribution later ranked at inference. Use physical synthetic mapped error
and deformation cost as feedback, and retain the parent-only matched baseline.
Do not use the inherited acquired Allen affine as hard anatomical truth for
pose. Evaluate all 14 branches, with selected and truth-best map error
separated. Expand exposure only while animal/plan-separated development
performance improves; retain the best checkpoint. Acquired-donor performance
is a transfer check with weak labels, not evidence of true alignment.

No probability calibration, electrode-region 90% claims, DeepSlice superiority,
or GUI default replacement follows from this protocol. Those require a stable
model, independent expert labels, strict animal-level held-out validation,
and an untouched final test later.
