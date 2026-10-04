# 088: reject nonmatching atlas correspondences (frozen before run)

## Question

The 086 fitted score barely ranked candidate planes (within-section rho 0.140 with physical error), and the fixed 087 correspondence-driven pose correction worsened all principal physical errors. In 083, a pixel outside the candidate's search window is excluded from correspondence training, while inference still forces one of 225/729 atlas cells. Can learning an explicit **no-match** class distinguish incorrect planes and damaged/background pixels from genuine anatomical matches? This is a targeted synthetic-development mechanism test, not a deployable or calibrated model.

The design transfers the unmatched-feature class from [SuperGlue](https://arxiv.org/abs/1911.11763) and the explicit missing-tissue alternatives in [Tward et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7027169/). Neither paper validates this particular histology-to-atlas implementation.

## Frozen training

Initialize the 085 batch-1,000 model/head and add one learned null logit after the 225 fine and 729 coarse depth–displacement classes. Keep the old image/atlas descriptor weights as initialization; train only those descriptors, their temperature and the null logit. Freeze the pose model, spatial updater, map, quality/ranking heads and atlas. No pretrained external features or model outputs are used as labels.

For each of 20,000 newly sampled eligible TRAIN sections (batch size one), construct the blind existing beam of eight old + six anchor branches. Select the physically best of these 14 by the preregistered 96-pixel rigid CCF distance plus 4,000 µm normal penalty, and one uniformly random distinct branch. Truth selects these two *training* branches and supplies correspondence labels; it never inserts a branch into the beam or enters inference. At each 32²/16² pixel, label the real atlas class only if the synthetic map is valid, inside the existing search window and has target-cell atlas support at least 0.5; otherwise label null. Balance mean cross-entropy of positive and null labels within each scale; count background as null at 0.1 weight. Average fine/coarse losses. Use AdamW at 1e-4 with cosine decay to 2e-5, save steps 0/2,000/10,000/20,000; retain draw provenance, exact source/parent hashes, label counts and loss curves. All inputs, computation and outputs remain on I:.

## Decision gate

After completion, audit raw training counts and loss/gradient trends and evaluate the frozen 061 synthetic DEV panel in one matched pass across the prespecified steps 0/2,000/10,000/20,000, using the same 14-branch beam and physical target as 086. Report positive/null correspondence error separately, matchability versus physical plane error within each section, selected and best-of-14 mapped CCF error, and black/raw/brush strata. A useful no-match head should improve plane evidence without sacrificing positive matching; a lower training loss alone is not enough. Do not tune a threshold or calibrate probability on the DEV panel. If this stage supplies meaningful signal, the next stage is a *joint* differentiable fit-to-pose training pass; 088 alone does not meet the goal's coupled-model requirement. Real held-out animals, final-test animals, electrode probability claims and public DeepSlice benchmarking remain untouched.
