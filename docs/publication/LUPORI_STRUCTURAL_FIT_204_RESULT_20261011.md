# 204: structural atlas fitting is locally informative, not a reliable global teacher

The frozen, no-update assay compared rigid finite-thickness Allen sections
with acquired Lupori fluorescence on identical eroded common-support pixels.
The diagnostic pair was selected from each model's *blind* top-16 by its
distance to the approximate VisuAlign reference: one within 1.5 mm and one
at least 3 mm away. The score did not see that reference. AL1A was withheld
from the 187 pose-training gradients; CC1B is the previously opened DEV animal
and was evaluated with the selected 188 all-TRAIN checkpoint. CAL, FINAL, and
public benchmark data were not read.

| animal / model | eligible pair | direct prior prefers near | intensity Pearson prefers near | MIND structure prefers near | permuted-intensity MIND prefers near |
| --- | ---: | ---: | ---: | ---: | ---: |
| AL1A / 187 | 37 | **26** | 24 | 22 | 21 |
| CC1B / 188 | 51 | 40 | 35 | **42** | 22 |

For a separate local ±250-µm AP perturbation of the near captured plane,
MIND preferred the physically closer rendering in 24/33 AL1A and 40/49
CC1B sections (Pearson: 22/33 and 35/49). This supports a conditional local
direction but does not rescue global plane selection: MIND is *worse* than
the direct prior on the withheld TRAIN animal and only 2/51 better on DEV.
These fractions are not population probabilities, and all references are
approximate human alignments from near-coronal physical sections. Historical
synthetic MIND fixed-pair success also failed when used as a blind search
objective (068/069). No fit score from 204 should be routed into global pose
probabilities or trained without a matched, guarded benefit test.

The first process wrote all section rows but failed while serializing a NumPy
integer in its summary, before creating a completion receipt. The corrected
`_retry1` run exited successfully; its config, 111 distinct source rows and
summary match `I:/AnatomyTracker/runs/lupori_structural_fit_204_guard_dev_001_retry1/completed.json`
(source SHA-256 `99a19ab986a27e109ba30b47ce1044f0d199ae0ed907d544d6cd7f2f41c73d47`).
The failed launch is not a separate scientific result.

The decision is to retain 188's direct probabilistic pose as the current
best acquired-image component, keep the deformation head experimental, and
seek a fit signal that survives both local and blind held-animal checks.
This is neither a qualified joint model nor evidence for arbitrary physical
oblique cuts, calibrated electrode probabilities, GUI use, or DeepSlice
superiority.
