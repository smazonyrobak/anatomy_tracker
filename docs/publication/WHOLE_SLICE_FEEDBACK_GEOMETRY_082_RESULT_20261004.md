# Whole-slice feedback 082: geometric capture and failure localization

The frozen 081 development gate failed because the selected synthetic
candidate worsened, while the best available candidate barely changed.
This read-only diagnostic asks whether exact synthetic tissue-to-CCF matches
were even inside 081's atlas search region. It does not use images to train,
select, or repair a prediction.

## Method

On all 246 eligible sections from eight independent 061 synthetic deformation
plans, use the frozen 081 final checkpoint and its exact 14 blind candidates.
At every query centre whose four neighboring source pixels are visible,
bilinearly average the synthetic ground-truth observed-pixel-to-CCF map.
Project that point into each candidate's physical frame. Mark capture if its
normal displacement is within ±6 mm and its lateral chart displacement is
within the actual fine ±2/32 or coarse ±8/32 window. Report the frozen 081
score-selected branch and the best *single* branch by ground-truth coverage.
The latter is a diagnostic oracle, never an inference choice. The run examined
41,785 visible fine-grid and 10,479 visible coarse-grid query centres. The
population comprises synthetic plans, **not independent biological animals**.

## Result

Plan-equal mean fraction of visible query centres whose exact 3D
correspondence is geometrically inside the search envelope:

| State and window | Frozen selected | Best single of 14 | Sections with ≥90% selected / best coverage |
| --- | ---: | ---: | ---: |
| Blind, fine | 48.0% | 83.0% | 77 / 173 of 246 |
| Blind, coarse | 81.3% | 98.3% | 135 / 239 of 246 |
| Two feedback passes, fine | 48.3% | 83.5% | 78 / 175 of 246 |
| Two feedback passes, coarse | **81.2%** | **98.3%** | **136 / 239 of 246** |

At the final selected state, normal-range containment is 94.7% on the coarse
grid; lateral containment is 83.8%; their joint containment is 81.2%. The
best single coarse candidate reaches 99.6% normal and 98.4% lateral, jointly
98.3%. That best candidate is one of the eight old branches on all 246
sections. The final score-selected branch is an anchor on only 3/246. On the
225 sections with at least ten visible coarse centres, the best-single coarse
mean remains 99.3% (223/225 sections ≥90%), versus 84.5% for the selected
branch (129/225 ≥90%). The conclusion is not driven solely by tiny sections.

## Interpretation and limits

The ±6 mm normal range is not the main geometric bottleneck on this panel.
The fine lateral window is too narrow for many selected candidates, but the
coarse search and at least one existing candidate usually contain the exact
match. Two learned feedback passes scarcely improve capture. Together with
081's unchanged best-of-14 mapped error and worse selected-minus-best gap,
this focuses the next experiment on **learning and ranking actual spatial
matches across the full candidate beam**, not merely widening the normal
range or running more batches of the same three-branch loss.

Geometric containment is necessary, not sufficient: it says nothing about
whether image and atlas features identify the correct match, the 1.5 mm depth
sampling resolves it, the atlas has useful local contrast, or the model can
propagate a coarse match into a precise pose and nonrigid map. A known target
outside the selected window can also be moved into range by changing the
candidate. This diagnostic uses synthetic ground truth only and makes no real
anatomy or calibrated uncertainty claim.

The implementation is
[`training/diagnose_whole_slice_feedback_geometry_082.py`](../../training/diagnose_whole_slice_feedback_geometry_082.py).
The I-drive output is
`I:\AnatomyTracker\runs\whole_slice_feedback_geometry_082`.
Its config records SHA-256 of the frozen 081 checkpoint, 061 panel records,
and 081 evaluation rows; independently re-hashing those three inputs matched
the recorded digests. All 246 diagnostic rows are present. The output summary
SHA-256 is
`6a87ab4a8022af1b54bb11ec3053928df008b5c70b5e54b369af1f68edee7265`.

This diagnosis is consistent with [DeepSlice's observation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10514056/)
that correct synthetic plane labels are valuable but anatomical ambiguity and
real-domain variation still matter. The target task is harder than its coronal
setting: any brain-intersecting plane, one section, and a local tissue map.
The [DISA histology-to-volume work](https://www.nature.com/articles/s41598-025-11583-w)
supports learning cross-modal similarity for initialization, but also starts
with a roughly known cutting axis; it does not establish full-angle accuracy
here. The literature motivates an explicit match-learning test, not a claim
that it will succeed.
