# 038 development diagnosis: how much atlas volume must a wrong-plane proposal see?

The frozen 019 direct predictor and the 177 eligible sections in the eight-identity, synthetic-only 037 development panel were re-used without training, image viewing, or touching the public/final tests. For each section, 512 evenly spread valid tissue pixels were mapped to their exact synthetic CCF coordinates. We decomposed the displacement from the frozen candidate's corresponding plane point into normal and in-plane distances. We report equal-identity means for its top-ranked candidate and the minimum-physical-error candidate among its top eight. This is a capture-volume calculation, **not** evidence that an image-to-volume matcher can identify the correct voxel.

| Frozen proposal | Mean 3D mapping error | Mean within ±2 mm normal and 2 mm tangent | Mean within ±4 mm normal and 4 mm tangent | Mean within ±6 mm normal and 6 mm tangent |
|---|---:|---:|---:|---:|
| Top-ranked | 2.500 mm | 62.0% | 81.9% | 93.1% |
| Oracle best of eight | 1.002 mm | 94.9% | 99.4% | 99.9% |

The top-ranked candidate has mean per-section 90th-percentile displacement of 2.23 mm normal and 3.19 mm tangent; the best-of-eight has 0.99 and 1.13 mm. Thus a single rendered plane at exactly the wrong candidate location omits substantial potentially matching anatomy. A local 3D atlas search spanning at least roughly ±6 mm along the candidate normal and comparable in-plane reach covers most synthetic tissue coordinates for this initial pilot, while a tighter ±2 mm search covers most oracle-candidate coordinates. Some sections lie outside even the broad window. The next experiment should expose an explicit atlas volume, train observed-pixel-to-3D correspondence on fresh TRAIN sections and compare true physical pose correction on this identity-disjoint development panel, including the imperfect-brush stratum and real donor regression gate. It must not be promoted from the geometric-coverage result alone.

Frozen output: `I:/AnatomyTracker/runs/atlas_3d_capture_diagnostic_038`; source: `training/diagnose_atlas_3d_capture_038.py`. The run's config records its source, parent checkpoint and panel manifest SHA-256; `completed.json` records the raw-row and summary SHA-256. No DeepSlice/public benchmark or animal-level final test was used.
