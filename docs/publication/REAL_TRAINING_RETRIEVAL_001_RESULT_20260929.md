# Real training retrieval: anchor/gallery mismatch, not simply donor overfit

The complete 256-image / 58-training-donor diagnosis exited successfully in
25.361 s. The failed8000 model matches its continuous affine anchor images much
better than whole A6000, but still retrieves the fixed catalogue poorly **on these
same training images**. The failed development gate remains failed. This is not
evidence that the problem is solely unseen-donor generalisation.

| Equal-donor training macro | A6000 | failed8000 |
|---|---:|---:|
| Own exact-affine anchor hit@1, among all256 | 48.02% | 85.60% |
| Any near-equivalent affine anchor hit@1 | 82.50% | 96.03% |
| Full-catalogue plane capture@32 | 15.43% | 22.59% |
| Full-catalogue plane capture@128 | 36.47% | 40.95% |
| Full-catalogue MAP normal error | 38.01° | 36.91° |
| Full-catalogue MAP normal-offset error | 3376 µm | 2859 µm |

Near-equivalent anchors mean antipodal normal≤10° **and** finite four-corner
RMS≤1mm, allowing the recorded horizontal raster correspondence. Plane capture
means normal≤10° and sign-aligned normal offset≤500µm. Neither definition is a
biological accuracy measure; both use the recorded weak upstream affine.
All full-gallery rankings/captures are unmasked. No six development donors,
new model inference, model fitting, channel selection or preprocessing tuning
were used in the subsequent analysis.

## Missing acquisition-frame nuisance

Every one of the 256 acquisition frames has **zero** near-equivalent cells in
the 98,304-cell catalogue. Even the closest finite frame has donor-macro RMS
2036.3µm; its range is 1316.1–2976.9µm. Its mean normal error is only 6.55° and
normal-offset error 259.3µm: geometric plane proximity does not make the rendered
finite image frame comparable.

For the nearest finite frame, the exact four-corner squared-error identity is
`||Δcentre||² + ||ΔU||²/4 + ||ΔV||²/4`, after choosing identity/horizontal U
correspondence and using finite pixel-centre edges 0..95. Each edge term separates
exactly into length difference squared and an orientation term. Pooled across all
256 rows, the squared error is:

| Contribution | Share |
|---|---:|
| In-plane finite-centre displacement | 48.93% |
| Through-plane finite-centre displacement | 2.06% |
| U/V span differences | 31.02% |
| U/V edge orientation differences | 17.99% |

These are squared-error shares, not additive RMS distances. Donor-macro tangent
centre displacement is 1374.8µm; normal displacement is 244.0µm. Recorded affine
full U/V spans average **12944.9 / 14093.4µm**, versus exactly **12000 / 12000µm**
in the catalogue. This is the atlas-space extent after the recorded affine, not
a change to the fixed 12mm acquisition-space preprocessing.

Shear is secondary: its half-edge magnitude averages 109.9µm. Removing the source
QR shear while retaining its finite centre and the same nearest cell changes mean RMS from 2036.3 to
2041.0µm. Shear and edge orientation couple; this counterfactual is not falsely
reported as a separate additive error share. The dominant missing nuisance is
in-plane centre and anisotropic span, followed by orientation, not shear alone.

Rows 19, 35 and 110 have no catalogue cell satisfying even the plane-capture
normal/offset limits. They remain in all-row results; missing eligible ranks are
NaN/null with counts, never an invented positive.

Restricting only this secondary geometric decomposition to the 253 rows that
have a plane-eligible candidate gives the same conclusion. The closest such frame
has RMS2054.0µm; pooled error shares are55.46% tangent centre,1.49% normal centre,
30.51% span and12.54% edge orientation. After minimally rotating the source normal
onto the candidate normal, the remaining absolute in-plane roll is only1.95°
donor-macro (normal tilt5.55°). Retaining source roll while fixing the key chart is
therefore a better-isolated first control than simultaneously randomising roll.

## Empty positives and actual-image inspection

The 11 zero-support anchors (rows 19,32,35,48,53,78,110,125,145,163,186) also have
**exactly zero atlas intensity**, while their real queries contain nonzero tissue.
Row205 has support mass44.672 and very little atlas content. Fixed previews retain
all12 plus the first12 other rows, at unchanged grayscale0..1. The latter show
recognisably corresponding image/atlas patterns; this visual check does not certify
the upstream affine or all rows.

For support≥64 (244 images, still all58 donors), failed8000 own-anchor hit@1 is
89.28%, near-anchor hit@1 is99.66%, but full-gallery capture@32 is only23.62%.
The gap is therefore not explained by the empty anchors alone. Among the11
identical empty anchors, hit@1 can reflect deterministic index tie-breaking;
failed8000's 1/11 own-anchor hit is not meaningful anatomical discrimination.
All these observations were preserved, not retrospectively filtered from training
or used to repair the frozen result.

## Exact chart construction for the next controlled bridge

The source catalogue uses `centre = support_origin + offset * normal`
(`training/arbitrary_plane_catalogue_v3.py`). A continuous affine anchor can use
that same chart **without quantising its plane or discarding its source roll**:

```text
c = O + (U + V)/2
u = U / ||U||
v = (V - (u·V)u) / ||V - (u·V)u||
n = u × v
d = n·(c - S)                         # S = recorded catalogue support origin
c0 = S + d*n                          # projection of S onto the exact source plane
U0 = 12000*u;  V0 = 12000*v
O0 = c0 - (U0 + V0)/2
```

This removes tangent translation, anisotropic span and shear from the **atlas key
chart**, retaining continuous normal/offset and the source U-axis roll. It does not
alter the query, invent a nearest-cell label, change the physical affine reference,
or imply segmentation. Its finite pixel47.5 centre is
`c0 - (U0+V0)/192`, not c0; c0 is the x=y=48 frame centre under x/96,y/96 rendering.
Any new key can still have insufficient anatomy and must be reported as such.
This is a motivated next control, not a demonstrated repair. Full-catalogue plane
coverage, held-out donor gates and eventual whole-model alignment remain required.

## Frozen evidence and reconstruction scope

- Diagnostic: `I:/AnatomyTracker/runs/joint_v6_real_training_retrieval_001`.
- Completion SHA256: `d1f83b33dbc030ca8bba422e1d8a1a93c72539ed88cfa34c8a8bf9c9ba12f782`.
- Saved diagnostic source SHA256: `966ff72065a43e02f91eb2b5425b9ba353f49b921b2fa3b7ad2d248298e5a605`.
- Analysis: `I:/AnatomyTracker/runs/joint_v6_real_training_retrieval_001_analysis`.
- Analysis summary SHA256: `5008f0891065677f5854bfbb2ea921352d74950f260bfedcad7534f081b59ee4`.
- Flat CPU analysis source: `I:/AnatomyTracker/runs/joint_v6_real_training_retrieval_001_analysis/analyze_real_training_retrieval_001.py`;
  SHA256 `6dcb4079e1c6adb3a0751ab238c40a3404934f095857ded90298583d1a1286ce`.
- Secondary plane-eligible geometry: `eligible_plane_decomposition.json` in the
  analysis directory; SHA256
  `b85795f9c9c46410dcbdef3ee958b5072b4d9e5f064a39d24975b40abc763891`.
  CPU source `I:/AnatomyTracker/runs/joint_v6_real_training_retrieval_001_analysis/decompose_real_training_eligible_plane.py`,
  SHA256 `d525bf04ccee53e762b2df1232ba4f61e889ff72c86e00a2e1e56e1bf2a29bbd`.

All15 files listed by the frozen diagnostic completion hash manifest matched.
Rendered channels and saved descriptors were finite. NumPy independently rebuilt
all256 exact-anchor score/rank comparisons and all256 full-gallery posteriors from
the stored descriptors and the two unchanged banks. Own/near-anchor ranks matched
exactly; maximum full-cell log-probability differences were9.09e-6 /7.40e-6 for
A6000/failed8000, with maximum log-normalisation error below4.88e-7.
This checks priors-once reconstruction, not new image encoding or atlas rendering.
FP64 finite-frame decomposition reproduced the saved nearest-frame RMS within
1e-7µm. Per-donor and support-stratified raw results remain available in the frozen
endpoint summaries/rows; no selected donor subset replaces the all-training result.

The evidence supports an acquisition-frame/gallery mismatch and a sampled-objective
closure failure. It does not establish that frame matching alone solves retrieval,
nor justify calling the failed model biologically accurate, calibrated, deployable,
or superior to a public comparator.
