# Whole-slice joint atlas feedback 081: frozen development protocol

## Why this experiment

On the frozen 061 development panel, the 059 model's 14-branch selected
visible-tissue CCF error is 2.656 mm, while its truth-best branch is 1.022 mm.
The pose candidate and its actual atlas fit therefore contain information that
the current selector fails to exploit. The 077 atlas-context updater trained on
physical pose correction alone did not materially improve selection; the 079
oracle broad-region reranker improved selection by only 0.329 mm and failed its
predeclared gate. The 075 direct-coordinate head and branch-ranking experiments
also did not improve the selected synthetic mapping. Repeating any of those
isolated mechanisms is not the 081 test.

The scientific analogy is specific, not an accuracy claim. [AMBIA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406728/)
localizes a mouse section in 3D and then deformably registers it to the extracted
atlas plane, but its two modules do not train the pose from the difficulty of
fitting that plane. [Iglesias et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6742511/)
show that uncertain registration and cross-modal synthesis can improve one
another through iterative inference, though their method does not solve our
arbitrary-plane single-section problem. [Chen et al.](https://www.nature.com/articles/s41598-025-11583-w)
use learned multimodal similarity for histology-to-volume pose initialization
followed by pose and deformation refinement; their study has a roughly known
cutting axis and is not evidence that our full-angle model will succeed.

## Frozen design

Start from the internally scratch-trained 059 joint checkpoint. A shared new
head sees each image feature and a 9-depth, finite-thickness atlas slab around
an actual blind pose candidate. Spatial 2D-to-3D correlations produce a bounded
full-frame pose update, one quality logit, and a spatial feature added to the
same model's local tissue-to-CCF mapper. After a first map, the head is reused
on its corrected pose with four fitting summaries: mean local warp magnitude,
roughness, atlas support, and correspondence reliability. The final mapping
error, fit-based ranking, and direct pose loss train the updater and, after a
500-batch head warmup, the pose and mapper weights jointly. The intermediate
summary is detached to fit the 11 GB GPU; the final fitting loss must have a
nonzero gradient into the parent pose head. This is an explicit memory-bounded
approximation to fully differentiating through both mapping passes.

Each batch uses one independently drawn synthetic section and one acquired
TRAIN section with a weak inherited Allen affine. Training chooses three
members of the *existing* 8-old/6-anchor blind candidate beam: highest-prior
old, highest-prior anchor, and the physically closest existing branch. Truth
is used only to choose a training example from that beam and supervise its
error; no truth plane is inserted at inference. The 3,000-batch pilot contains
3,000 fresh synthetic presentations and 3,000 acquired TRAIN presentations,
unique within this run but not necessarily across earlier runs. Every accepted
and rejected synthetic draw and every acquired image preserves its source IDs.
The synthetic target is known observed-pixel-to-CCF correspondence; the acquired
affine is weak, not a biological ground truth. Only internally trained weights
are used; no external model features, checkpoints, or pseudo-labels enter.

At inference, process all 14 blind candidates in chunks of two with two shared
updates and fit summaries, then select the final atlas-conditioned mapped
branch. No user mask or automatic segmentation is required. Raw background,
black exterior, and imperfect brush input are variations in the synthetic
stream, never deliberately paired copies of a section.

## Readout and gate

After the runner exits, verify checkpoint and source hashes, then evaluate the
frozen steps 0, 500, 1,500, and 3,000 on the same 246 eligible 061 synthetic
DEV sections from eight independent *synthetic deformation plans* and 64 weak-
real DEV sections from six donors. The latter donors are separate from acquired
TRAIN; synthetic plans are not biological animals. Parent 059 is measured on
the same 14-candidate beam. Use identity-equal means and retain every section.
No public benchmark or final-test animals are touched.

The pilot advances only if its final checkpoint improves selected synthetic
visible-tissue CCF error by at least 0.400 mm, reaches at most 0.900 mm
truth-best-of-14 mapped error, and worsens the donor-equal weak-real mapped
error by no more than 0.100 mm. Report both selected and truth-best errors so a
ranking gain cannot conceal failed pose capture. Passing this *development*
gate is not a calibrated probability claim, external validation, DeepSlice
superiority, or permission to replace the GUI default. The best observed
checkpoint is retained for the next training decision, not selected on a final
test.
