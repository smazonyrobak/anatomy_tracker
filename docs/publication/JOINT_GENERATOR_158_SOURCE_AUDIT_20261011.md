# Source audit of the 158 training images

The 158 comparison calls `sample_panel_geometry_train_151` and then the v3/v4
artifact function with that result as `source_sample`. It does **not** call the
older v7 appearance-v3 geometry sampler. This distinction matters when
interpreting 158 and planning the next training stream.

- Every attempted image gets an independently randomized plane normal, roll,
  through-plane thickness, offset, reflection, appearance and artifacts. The
  two experimental arms intentionally replay the same accepted draws for a
  matched comparison; there are no paired background variants or deliberate
  rejection of coincidentally similar planes within an arm.
- Normal sampling is isotropic up to antipodal equivalence and roll is uniform,
  so steep and cardinal cutting angles are supported. The offset is uniform
  across the projected **subject box**, not uniformly over informative brain
  intersections. Training accepts only sufficient visible tissue and at least
  four valid 24×24 query sites. Small/grazing sections remain difficult and
  some are rejected; this is not evidence of exhaustive all-angle accuracy.
- The 158 script chooses from 60 fixed TRAIN deformation maps. The loader also
  provides virtual global-affine variants of those maps, but the actual 151
  sampler indexes `context['bases']`, not `context['subjects']`, so those variants
  do not diversify this run. Nor does 151 use the older sampler's support-point
  offset or TRAIN acquisition-framing branches. This is a concrete source-level
  distribution restriction, not a posthoc model-performance explanation.
- V3 samples a smooth spatial warp and independent fragment, fold, bubble and
  tile-seam flags. The implementations are a single roughly elliptical moved
  fragment, a narrow straight fold, a circular lens/blur bubble, and one
  axis-aligned half-image seam. V4 adds exposure and tissue-only read noise.
  The source image still comes from one atlas and a low-frequency texture field;
  these named events are **not** validated physical slide simulators.
- V3 favors raw/no-brush input and, conditional on raw, an exact-black exterior
  80% of the time. Gray/confusing backgrounds remain possible, but their
  prevalence has not been measured for the intended laboratory. An earlier
  donor-equal [TRAIN appearance audit](REAL_SYNTHETIC_APPEARANCE_GAP_20261009.md)
  found a border mismatch for the older v2 distribution; it did not establish
  that v3/v4 reproduces tissue texture, defects, or scanner artifacts.

**Next data correction, before scaling exposure:** version a new sampler that
mixes independently drawn support-point and full-box plane offsets, uses
acquisition-like crop/scale/shear framing, and samples the existing virtual
variants without calling them independent animals. Preserve full-angle
normals, the marginal-section stress branch, exact provenance and one fresh
physical section per presentation. In parallel, use only TRAIN physical
slides to estimate joint tissue/background texture, focus and defect severity;
then broaden the synthetic shapes and test a small fresh unpaired QC panel.
Do not add paired recolorings or infer physical prevalence from event names.

This audit reads source only. It does not evaluate the active 158 output, make
a model-performance claim, or relax the need for animal-separated acquired
oblique validation and calibrated uncertainty.
