# Opt-in rendered-atlas image retrieval

Status: untrained primitive only. No training driver, experiment, GPU execution,
or global-capture/calibration claim accompanies this change. Existing forwards
and default model construction are unchanged.

## Implemented boundary

`image_key_descriptor_dim=None` creates no descriptor parameters. Setting it to
256 adds one shared `Linear(F*4*4,256)` after spatial 4x4 adaptive pooling and
L2 normalization of its output. `descriptor_from_features` accepts features
from the current histology or atlas stem and shared encoder; it does not load
another encoder. Optional initialization uses an isolated CPU RNG scope, so
existing parameter draws and subsequent initialization remain unchanged.

`image_key_cosine_logits(query[B,D], bank[N,R,D], temperature)` returns only
unnormalized, temperature-scaled cosine scores `[B,N,R]`. It does not cache,
detach, collapse representation keys, add priors, or replace the current
proposal forward. Gradients reach both descriptors and their shared projection.

For a truly complete bank, a future caller must add log representation priors,
log-sum the representation scores, add bound cell log mass, and normalize over
all catalogue cells. Representation conditionals remain separately available.
A softmax over sampled negatives is not the full-catalogue posterior and is
not calibrated uncertainty. Horizontal raster reflection must remain distinct
from anatomical ML reflection; opposite electrode locations must not be averaged.

## Bounded next experiment, after joint rehearsal

- Initialize one fresh whole joint model and descriptor projection; no old
  encoder merge, model weights, external features, or pseudolabels. Retain the
  current recurrent pose/deformation components as the eventual native updater.
- Train 4,000 proposal-retrieval updates with the current 8-generated/8-frozen
  query composition and source observations. Compare against original003 at
  step 4,000, including actual applied updates, presentations and compute cost.
  This compares proposal mechanisms, not a single-cause capacity ablation.
- Encode positive atlas renders and bounded globally sampled/nearby negatives
  using the same current shared encoder. Train a multi-positive contrastive
  objective; do not make neighboring/equivalent poses hard false negatives.
  Keep low-support censoring, full brain-intersecting-plane sampling and exact
  original/synthetic provenance. No automatic segmentation is required.
- Use 96x96 nine-sample finite-thickness atlas renders and fresh differentiable
  keys during training, with small checkpointed encoder chunks. No stale atlas
  feature cache while encoder parameters change. Memory feasibility on the
  11 GB 2080 Ti remains to be measured, not asserted by this source-only change.
- At frozen endpoints, rebuild the complete 98,304-cell bank with both raster
  representations. The first gallery has a declared 50 um PSF; query thickness
  variation therefore produces a known acquisition mismatch. Bind any bank
  receipt to weights, catalogue, PSF and representation convention. A 256D FP16
  two-representation descriptor bank alone occupies 96 MiB.
- Record full-catalogue NLL, normal marginal/MAP, physical offsets, top32/128
  geometric capture, reflection accuracy and all brush/censor subsets. Proposed
  promotion: at least 5 degrees lower normal error and improved top32 physical
  capture versus the matched baseline, without a material brush-mode regression.
  Thresholds must be frozen before launch. No public benchmark at this stage.

These are organizational synthetic groups from one atlas, not independent
biological animals. Real acquired backgrounds, coherent subject variation,
strict unseen-animal validation and subsequent uncertainty calibration remain
necessary. Better sampled contrastive accuracy alone does not satisfy them.

## Scientific basis and limits

[CoMIR](https://proceedings.neurips.cc/paper/2020/file/d6428eecbe0f7dff83fc607c5044b2b9-Paper.pdf)
demonstrates contrastive learning of dense, spatially equivariant multimodal
representations for registration. It does not establish arbitrary-plane global
histology retrieval. Our descriptor proposal is an adaptation, not a reproduction.

[Synth-by-Reg](https://pmc.ncbi.nlm.nih.gov/articles/PMC8582976/) supports
geometry-preserving contrastive supervision for histology/MRI, but assumes
paired corresponding planes and a previously trained registration network.
Neither its weights nor a known test-plane correspondence is imported here.

[SVoRT](https://arxiv.org/html/2206.10802) motivates synthetic orientation
coverage and iterative image-informed transform updates. Its multi-stack MRI
context does not remove the ambiguity of an isolated histological section.
