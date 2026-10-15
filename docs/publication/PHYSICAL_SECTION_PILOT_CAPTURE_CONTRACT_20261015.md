# Physical section pilot capture contract

The next useful data are **physical sections whose cutting planes can be located independently of this model**. Existing Allen section affines and virtual oblique reslices do not provide that check. Start with a small TRAIN-only feasibility tranche, not the full qualification cohort: two distinct animals, one cut near a cardinal plane and one whose plane normal is at least 30° from each cardinal-plane normal (ignoring normal sign), with about 8–12 sections per animal across different anatomical levels. These counts test the capture and coordinate chain; they cannot establish all-angle accuracy or calibrate probabilities.

## Record at acquisition

- Give every raw section one immutable animal, specimen, experiment, block and section ID. Preserve cutting order, intended and *measured* cut direction, thickness, missing sections, staining/imaging method and pixel scale. A probe insertion angle is not a cutting-plane angle.
- Keep the original full slide or stitched image with its acquired background and defects. Save any smart-brush outline as a separate optional input; do not manufacture a black-background twin of the same section.
- Obtain a same-animal intact 3-D scan before sectioning, or a calibrated block-face photograph immediately before each cut. For block-face imaging, retain scale/fiducials, camera geometry and section order so the photographs form a specimen-space stack. The [block-face histology method](https://pmc.ncbi.nlm.nih.gov/articles/PMC4753455/) explains why pre-cut photographs can anchor sections despite processing deformation. Preserve all raw images, transforms and coordinate conventions on `I:`.

## Establish usable reference geometry

Map raw section pixel centres to the cut surface in specimen space, then map that specimen frame to the pinned Allen CCF. Record the complete signed axes, units, handedness, origins, voxel-centre convention and transform direction at each step; a scale-only header or a plausible tissue overlay is insufficient. Check the chain on asymmetric left/right anatomy and on cortex, ventricles, hippocampus and deep structures at several section levels. Have an expert mark homologous landmarks without seeing model predictions, and retain per-landmark residuals and any unidentifiable or missing tissue. The resulting *physical cut plane*, *specimen-to-CCF registration* and *local histology distortion* have separate uncertainty; do not collapse them into an exact label.

## Gate the next model run

Use the pilot only after those coordinate and landmark checks pass and its animal IDs are absent from reserved development, calibration and final cohorts. Compare the frozen image-only model with and without the **measured sectioning direction**, then test whether atlas-plane fitting actually improves blind plane choice and surviving-tissue mapping on these same physical sections. Report every case, including failures and low-information fragments. If the reference is too uncertain or the model still cannot distinguish plausible planes, improve the acquisition/reference chain or return broad uncertainty; do not scale synthetic training or claim a 90% electrode-region probability from this pilot.

Only after a working TRAIN pilot should new animals be collected across cardinal and steep-oblique directions, with whole animals prospectively assigned to development, calibration and untouched final testing. DeepSlice comparison and GUI replacement remain later gates, not substitutes for independent physical-plane truth.
