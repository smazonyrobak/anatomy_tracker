# 130: does anatomical source–atlas interaction improve blind plane selection?

The frozen 128 treatment learned the auxiliary correspondence labels but did
not improve blind plane capture. A post-exit 2×2 diagnostic paired 43 disjoint
DEV section pairs with similar atlas support. The existing matcher had mean
source×atlas interaction 0.0253 with full atlas intensity and 0.0239 with that
intensity zeroed; only 6/43 pairs increased absolute interaction by at least
0.01 when atlas intensity was present. This is conditional mechanism evidence,
not an accuracy estimate: the old scalar was trained as correction propensity,
and its head pools beyond the common intact tissue. It does not justify
fitting-to-pose feedback yet.
The read-only diagnostic JSON is
`I:/AnatomyTracker/runs/appearance_gap_129/crossed_image_atlas_128.json`
(SHA-256 `c12e717c1cf2b09b091c0f670fd8b781594ef57946059a59fd56256962aca107`).

The next experiment retains the 128 model's randomly initialized lineage,
full arbitrary-plane state, atlas renderer and local mapper. It adds one small
evidence head to the existing global matcher. The old score only changed the
corrected action. The new evidence is added to the *original and corrected*
branch scores; the old correction logit remains separate. In this first stage,
the direct pose and mapper are frozen, and selection uses original candidates
to isolate whether atlas-conditioned evidence can rank them. A later stage may
jointly train pose and mapping only if this evidence is useful. This is a
staged part of the intended model, not a separately pretrained component or
a claim that the full feedback requirement is already met.

Two matched 2,000-batch arms each see two independently sampled physical v4
synthetic sections per batch. The full arm receives atlas intensity, support
and coordinates; the control receives support and coordinates, with atlas
intensity zeroed. Both start from exactly the same
128 checkpoint and same-seed newly initialized evidence head, use the same
seed formula, loss weights and optimizer, and record every accepted draw's
provenance. All sample plane, appearance, background and damage settings are
drawn independently; there are no systematic paired versions of one section.
No external weights, legacy pseudolabels or public benchmark labels enter.

The training objective has two parts:

1. For two independently drawn sections A/B, render their known atlas planes
   and score the 2×2 source–atlas combinations. Increase the matched-versus-
   crossed interaction `E(A,A)+E(B,B)-E(A,B)-E(B,A)`. Additive source-only and
   atlas-only biases cancel in this difference. Shape/support shortcuts are
   controlled by the matched support-only arm, not assumed absent.
2. Where a blind-16 beam contains a near rigid candidate and a wrong candidate
   with similar atlas support and common intact tissue, increase the posterior
   margin of near over wrong. Synthetic truth is used to construct TRAIN labels
   only; no truth mask or location is passed to inference.

The contrastive cross-modal idea is informed by [CoMIR](https://papers.neurips.cc/paper_files/paper/2020/file/d6428eecbe0f7dff83fc607c5044b2b9-Paper.pdf)
and [ContraReg](https://pmc.ncbi.nlm.nih.gov/articles/PMC10415941/), including
ContraReg's warning that empty background can create false patch pairs. These
papers do not establish that the present histology task is solved by this loss.

Evaluate frozen steps 500, 1,000 and 2,000 only after both arms finish. Use
the same 243-section synthetic DEV panel and frozen original-branch mapped
errors as the 128 result, so branch decisions are directly comparable.
The 151 pre-existing *original-branch* support-matched near/wrong pairs are
the primary conditional anatomy test. The corrected-action pair set is not
interchangeable with this original-branch cohort. A promising evidence signal
must reach at least
75% pair wins and exceed the support-only arm by at least ten percentage
points; zeroing atlas intensity and swapping source features should each
reduce its pair-win rate by at least ten points. For localization rather than
pair classification alone, plan-equal selected mapped error should improve by
at least 0.30 mm and within-1.5-mm capture by ten points against both the
parent prior and support-only control. Preserve the absolute result and plan-
level harms even if a threshold is missed. The blind-beam prior baseline is
the highest frozen input score among those same candidates, not beam slot
zero; also report the globally prior-ranked branch separately. A
post-training 2×2 check on the
support-matched pairs and a fresh v4 panel are needed before claiming a robust
mechanism; the reused panel is development data, not independent animals.

Only if this synthetic gate is promising should the project check acquired
donor-level weak-reference guardrails and then connect this evidence to the
direct probabilistic pose head with a matched feedback/no-feedback experiment.
The intensity-zeroed control inherits a matcher previously trained with atlas
intensity. It is a matched-input ablation, not a clean from-scratch estimate
of the causal value of anatomy; source-swap and atlas-zero results must be
considered alongside it. Both arms change the shared hidden head, so the old
corrected-action logit and pose update are not preserved even though their
last linear layer is frozen; this stage evaluates original branches only.
The crossed interaction also uses source-row-specific thickness in both atlas
columns. This avoids a diagonal-only thickness cue, but a fixed/shared-PSF
post-training check is needed before claiming the interaction is purely
anatomical. Verify identical source hashes and accepted draws across arms
before applying any comparison gate. The inherited Allen alignments are not
blinded expert truth. No calibration,
GUI promotion, DeepSlice comparison or arbitrary-oblique performance claim is
authorized by this stage alone.
