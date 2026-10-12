# V9 generator audit 285 protocol (frozen 2026-10-12)

## Question and scope

This TRAIN-only audit measures two properties of the unchanged v9 synthetic generator:

1. whether its proposed arbitrary planes remain represented after tissue-support eligibility, with enough valid sites for the 2,048-site dense target, and whether requested slide artifacts are actually targeted onto tissue;
2. how far the saved pre-artifact affine pose gauge lies from a robust affine fit to the final valid, intact pixel-to-CCF coordinate field.

It does not train or select a model, use DEV/CAL/FINAL animals, use real data, or establish biological accuracy.

## Frozen draw

- Generator: `sample_one_shot_slide_artifacts_v9`, unmodified, side 192.
- Exactly 512 proposals: eight independently drawn virtual subjects from each of all 64 TRAIN deformation bases.
- The eight variants within a base are sampled without replacement. Each row is generated once with one appearance; there are no retries and no repeated physical section with a changed appearance.
- Generator `eligible` is the accepted population. Counts of `min(valid_pixels, 2048)` are reported separately, including the fractions reaching 128 and 2,048 sites.
- Proposal and accepted coverage report sampling branch, nearest-cardinal normal angle, absolute normal components, visible support, valid-site counts, and per-base acceptance.
- For fragment, fold, bubble, and tile-seam events, an on-tissue realization means that the event fired and the generator's explicit `tissue_placed` flag is true. Returned affected-pixel counts are retained separately where available.

## Frozen affine gauge fit

For each section, fit CCF AP/DV/ML coordinates from the final `centre` field using pixels satisfying
`valid_mask & ~fragment_displaced_mask`. Thus displaced fragments are excluded; folds, severe bubble blur, background, overlap, and out-of-frame pixels are already excluded by `valid_mask`.

The design is `[1, x, y]`, with `x=(191/192)-x_observed` when the saved horizontal-reflection flag is true and ordinary `x_observed` otherwise. Raster coordinates are divided by 192, matching the generator state convention. Start with ordinary least squares, then perform exactly three Huber IRLS updates. At each update, residual is Euclidean CCF error in micrometres, scale is `max(1.4826 * MAD(residual norm), 25 um)`, cutoff is `1.345 * scale`, and weight is `min(1, cutoff / residual)`.

The audit records saved-state and robust-fit residuals on the same intact pixels, their mean pointwise plane separation, centre separation at `(0.5, 0.5)`, unsigned plane-normal angle, coefficients, and fit condition number. It also retains the initial ordinary-least-squares decomposition of target displacement from the saved plane into the support-restricted `[1, x, y]` affine subspace removed by the mapper's `project_surface_affine_out` operation and its orthogonal local residual. It reports RMS magnitudes and the exact squared-error fraction in that affine component. This valid-intact support projection is the measurable analogue of the mapper's full-canvas projection; it is not claimed to be numerically identical on unobserved pixels.

The v9 pose-label stage is audited from frozen source/provenance: v9 returns the state inherited unchanged through v7/v6/v4/v3, while v3 explicitly describes it as the source affine gauge fitted before v3 artifacts. Therefore the saved pose is a pre-artifact label, not a post-artifact refit. The empirical gauge/decomposition measurements quantify the consequence rather than relabeling it.

## Outputs and interpretation

The run writes only under `I:/AnatomyTracker/reports/v9_generator_audit_285_train_001`: frozen config, complete generator context provenance, one JSON row per proposal, aggregate summary, and a completion receipt with SHA-256 hashes. Source and protocol are committed before execution. Results are descriptive: any strong proposal-to-accepted orientation shift, material 2,048-site shortfall, low realized-on-tissue rate, or large saved-versus-fit drift is a generator/supervision limitation to address in a later experiment, not silently corrected here.
