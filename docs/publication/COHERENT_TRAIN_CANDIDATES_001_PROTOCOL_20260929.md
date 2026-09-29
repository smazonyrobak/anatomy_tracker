# Image-ranked coherent TRAIN candidates

Prepared prerequisite for sequential native training; no optimizer or native
refinement, no development/benchmark cases and no model qualification gate.

The completed TRAIN frame analysis showed that the previous truth-near local
initializer misses actual retrieved-frame variation. Obtain actual image-ranked
starts on the coherent corpus before choosing its capture curriculum.

- Use the whole audited coarse A22576 checkpoint and its matching98304-cell,
  two-reflection gallery. No independently trained refiner or encoder is merged.
- Include all512 base coherent TRAIN sections and128 acquired-view TRAIN
  sections, each with raw, exact-black and imperfect-brush presentations:
  640 physical sections/1920 observations, the same eight synthetic subjects.
  Retain every censored or empty observation. Never open held-out section files.
- Query inputs are only each saved image, optional outline and its availability.
  Subject IDs, fitted planes, support/visibility truth, deformation fields and
  true PSF parameters cannot enter retrieval or candidate selection.
- Use the established FP32/TF32-off cosine/.1 score, representation prior,
  representation marginalization and cell mass exactly once. Save stable
  image-ranked top128 cells, both reflection probabilities/full-catalogue
  component masses, query descriptors and retained/omitted coarse mass.
  Cached descriptors are for reconstruction, not training targets.
- The gallery's fixed50um PSF is unchanged; the coherent observations have their
  own saved finite-thickness acquisition. This is a declared mismatch, not a
  claim that the query's true thickness is available at inference.
- Preserve canonical fitted frame, observed reflection, exact source/observation
  IDs, source artifact hashes and supervision eligibility separately. Geometry
  is for TRAIN supervision/readout only; no target-selected start substitutes
  for the saved image ranking. Acquired-view nuisance-donor IDs remain separate
  from synthetic anatomy identity. This supplies no additional biological brains.
- Bind the exact whole checkpoint, gallery, catalogue, source and both corpora.
  Use no optimizer, no rerender and no source modification to the live normalized
  native experiment. Future encoder or gallery-contract changes invalidate this
  cache; later joint training must use fresh features from its own whole model.

Readouts distinguish base/acquired views and modes, eligibility/censoring,
antipodal plane error/capture and reflection-aware finite-frame error. Probabilities
remain uncalibrated and any oracle beam-capacity readout is explicitly labelled.
Do not revise the ongoing normalized-native experiment from these results.

Checkpoint SHA256:
`e22ff8e13a09b8c624c64518ca65a0ac9feda2c823dde30b6b8b5707995d52d8`.
Gallery SHA256:
`5219117c897bf68fb6139b5bd482bfdd05f5c52e2b9a5a7296c9936d0f8d0b36`.
Coarse independent audit:
`c75c4ff817e7603eca31c8086080aafdb7f0953b3fec58c3f3f2f7842cf2afb3`.
