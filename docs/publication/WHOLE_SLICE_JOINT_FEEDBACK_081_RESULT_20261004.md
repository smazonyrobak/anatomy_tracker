# Whole-slice joint atlas feedback 081: development result

## Decision

**Fail the frozen development gate. Do not promote 081 to the GUI or run a
public DeepSlice benchmark on it.** The final 3,000-batch checkpoint worsened
selected synthetic visible-tissue CCF error from 2.535 to 2.792 mm on the
matched panel. The best of its 14 mapped candidates remained 0.988 mm, above
the 0.900 mm requirement. It did improve agreement with the inherited weak
Allen affine on acquired sections, but that affine is not verified anatomical
truth. This is not a usable accuracy or uncertainty claim.

The [frozen protocol](WHOLE_SLICE_JOINT_FEEDBACK_081_PROTOCOL_20261004.md)
specified these gates before evaluation. A clarification there distinguishes
the historical 061 **prior-selected rigid** 2.656 mm from the 081 evaluation's
matched 059 **fitted-selected mapped** 2.535 mm. The two endpoints must not be
substituted for one another.

## Matched development comparison

Mean physical CCF error in mm; synthetic means first average sections within
each of eight independent synthetic deformation plans and then average plans.
The real column averages six independent DEV donor means. All four checkpoints
use the same 246 synthetic sections, 64 acquired DEV sections, and blind
14-candidate beam. The selected map is measured at 256 pixels; best-of-14
is a truth-informed diagnostic measured at 96 pixels, not an inference output.

| Head batch | Synthetic selected map | Synthetic best-of-14 map | Acquired weak-affine agreement |
| ---: | ---: | ---: | ---: |
| Parent 059 | 2.535 | 0.986 | 1.138 |
| 0 | 2.522 | 0.992 | 1.242 |
| 500 | 2.594 | 0.997 | 1.285 |
| 1,500 | 2.893 | 1.044 | 0.814 |
| 3,000 | **2.792** | **0.988** | **0.745** |

The final synthetic selected gain requirement was at least 0.400 mm. The
observed change is **0.256 mm worse**. The best-of-14 requirement was at most
0.900 mm; observed is 0.988 mm. The donor-equal weak-real requirement allowed
at most 0.100 mm worsening; observed is 0.393 mm improvement against the weak
affine. The first two gates fail, so the conjunctive gate fails.

## Where it failed

The selected branch changed on 122/246 synthetic sections. On those changed
sections, the section-weighted mean error rose from 2.456 to 2.976 mm. On the
124 sections retaining the same branch, it moved only from 2.619 to 2.604 mm.
These subset means diagnose branch selection; they are not the identity-equal
primary endpoint. At the final checkpoint, selected error worsened on exact
black exteriors (2.322 to 2.835 mm) and imperfect-brush exteriors (2.557 to
2.928 mm), while raw backgrounds improved (2.722 to 2.524 mm). Low-support
sections worsened from 3.537 to 3.862 mm. Thus the regression is not a mere
black-background-versus-raw artifact, and the new correlation/feedback head
has not learned a robust whole-slice ranking signal.

For the acquired weak-affine cases, the parent selected an anchor branch on
63/64 sections and the final head on only 3/64; no section retained the same
selected branch. The large weak-affine improvement can therefore be explained
in substantial part by a branch-selection shift toward the old near-coronal
bank. Without accurate independent biological plane labels, it is not proof
of better anatomical localization. The head's physical pose correction also
barely changes the synthetic best-of-14 endpoint. More batches of this same
objective are not justified by these results.

## Integrity and scope

The 3,000-batch run made 3,000 fresh synthetic and 3,000 weak-real TRAIN
presentations. The training receipt's config, draw ledger, training log, and
four checkpoint hashes were independently checked; so were the development
receipt's config, 1,240 per-section rows, and summary hashes. The frozen
summary SHA-256 is
`dbb268f7f47e398bb2cfdb296900c17d1a7b94a3d1a0d2dccd54b9d37292d249`;
the rows SHA-256 is
`42d4a81a78e40281b410f17afef652b92b15b714cfe9517a7a7c2ff798c76365`.
The final checkpoint SHA-256 is
`e24d8b1edac96bcabf5688dc63776a63b4d89fe315ebf7228c4fdc3eb4c6bfc2`.
Full receipts and raw rows remain under
`I:\AnatomyTracker\runs\whole_slice_joint_feedback_081_pilot` and
`I:\AnatomyTracker\runs\whole_slice_joint_feedback_081_development_eval`.
Both receipts declare `calibrated: false` and `public_benchmark_used: false`.
The synthetic plans are not biological animals. Acquired DEV donors are
separate from acquired TRAIN; untouched final-test animals remain untouched.

## Next discriminating experiment

Before another training run, measure whether the *known* synthetic
observed-pixel-to-CCF correspondences lie inside 081's actual local 2D-to-3D
atlas search region for selected and truth-best blind candidates. If not, a
wider or hierarchical capture stage is required; a feature loss cannot create
matches outside the window. If they do, supervise dense correspondence
directly with the known synthetic map, then revisit feedback ranking. This is
an inference from 081's failure pattern, not an established mechanism.
