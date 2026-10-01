# Deferred validation and uncertainty plan

This plan is deliberately deferred until the cold-start method and training
regimen stabilize. Early architecture work must preserve the fields needed to
execute it, but must not repeatedly inspect final-test animals.

## Data custody and splits

- Preserve source dataset, experiment/laboratory, animal/specimen and section
  identifiers in every manifest, prediction and exclusion record.
- Split by animal before any view generation. Augmented descendants and serial
  sections from one animal stay in one split.
- Treat all sections and views sharing one synthetic 3-D animal warp as one
  pseudo-animal group; never split synthetic descendants by slice.
- Freeze untouched final-test animals after architecture, losses, thresholds
  and calibration methods are selected.
- Use Allen/CCF-derived synthetic views and eligible Allen sections for
  training. Use the public [DeepSlice Source Data and Ground Truth dataset
  (10.25949/22802411.v1)](https://doi.org/10.25949/22802411.v1) as a transparent
  public benchmark, not as the untouched final test, because existing project
  development has already inspected related DeepSlice cohorts.
- Prefer a separately collected multi-laboratory real-histology cohort for
  external validation.

## Reference and comparison

- Obtain blinded independent expert alignments, retain individual raters and a
  prespecified consensus rule, and record correction time.
- Compare identical raw cases against frozen DeepSlice modes, relevant
  expert-assisted tools and the frozen legacy pipeline under clearly separated
  automatic and assisted tracks.
- Report the proposed method without a user mask as the primary automatic
  track. Report one frozen automatic-outline sensitivity track and a separate
  smart-brush-assisted track; never give only the proposed method a manually
  corrected outline in an automatic comparator claim.
- Predefine physical landmark registration error as the unique primary
  anatomical endpoint. Treat corresponding-plane distance, geodesic normal,
  physical offset and in-plane-frame error as secondary pose-track metrics
  across the full plane domain; retain
  AP/L--R/D--V summaries only for well-conditioned coronal/applicable
  comparator subsets. Also report regional overlap and boundaries, topology,
  failures, abstentions and human correction time.
- Treat animals as the statistical units. Report paired effect sizes and 95%
  confidence intervals, with all attempted cases and failures retained.

## Probabilistic pose and downstream uncertainty

The experimental one-shot model predicts multiple full-frame pose components with
reflection scores and component scales. These are **not** calibrated posterior
probabilities; its dense-map covariance output is not yet trained. Any later
atlas-fit reranking changes the effective distribution and must be evaluated
and calibrated as part of the complete model. Constraints may condition a
calibrated posterior, but cannot manufacture certainty after inference.

Pose, deformation and between-section/within-animal dependence must all be
represented before propagating uncertainty to a trajectory or brain-region
assignment. The current outputs are insufficient for those claims and remain
explicitly uncalibrated.

Fit any temperature or conformal calibration only on held-out calibration
animals. On unseen animals report negative log likelihood/proper scoring rules,
risk--coverage, interval width, and empirical 50/80/90/95% coverage with animal-
level confidence intervals. In particular, a nominal 90% pose region should
contain the blinded reference about 90% of the time. Probabilistic output is
eligible only if point-estimate accuracy is noninferior to the matched
deterministic head.

Only after normal/offset, in-plane frame and dense-map uncertainty are all
represented and calibrated may their joint samples propagate through the probe
solver to produce a centre trajectory, credible spatial volume and per-region
assignment probabilities. The GUI must label uncalibrated development scores
as compatibility/risk, never as confidence.

## Reproducibility receipt

Freeze and retain exact animal splits, raw input hashes, code commit, atlas and
ontology versions, configuration, seeds, selected checkpoint/export hashes,
environment, per-case raw predictions, failures, exclusions and statistical
scripts. The detailed protocol and power analysis are refined from current
peer-reviewed practice only after the method stabilizes and before final-test
access.
