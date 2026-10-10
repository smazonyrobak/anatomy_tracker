# 131b: fresh draw confirms an exposure-sensitive direct-pose failure

The frozen 128 treatment model completed the predeclared confirmation without
updates: 256 eligible, independently drawn v3 physical sections and 256 v4
sections, four per each of 64 TRAIN synthetic local-deformation subjects in
each cohort. The 568 attempts, 512 accepted section IDs and 512 plane hashes
were audited; there was no deliberate same-plane appearance pairing. Exact
128 blind-16 proposal construction and observed-valid-pixel rigid-gauge
distance were used. All output/source/checkpoint/config hash bindings passed.

| Appearance | Best beam mean | Best beam within 1.5 mm | Direct top-one mean | Direct top-one within 1.5 mm |
| --- | ---: | ---: | ---: | ---: |
| v3, 256 sections | 1.057 mm | 210/256 (82.0%) | 2.586 mm | 108/256 (42.2%) |
| v4, 256 sections | 1.701 mm | 135/256 (52.7%) | 3.425 mm | 61/256 (23.8%) |

The beam-capture difference is 29.3 percentage points. A 20,000-resample
bootstrap over the **64 subject bases**, keeping all four sections together,
gave a 95% percentile interval of 21.5–37.1 points. The v3 and v4 mean
nearest-cardinal angles were 31.0° and 33.0°; within the 15–30°, 30–45° and
45–55° strata, v3/v4 beam capture was 82.2/55.9%, 84.3/47.3% and 84.0/59.4%.
The 0–15° stratum was 71.0/64.0% and smaller. These independently sampled
cohorts are not a matched causal contrast, but the size, sign and within-angle
pattern confirm the synthetic-domain failure.

In v4, beam capture was 16/79 (20.3%) for exposure <0.15, 19/51 (37.3%)
for 0.15–0.30, 55/70 (78.6%) for 0.30–0.60, and 45/56 (80.4%) for
0.60–1.20. The high-exposure (≥0.6) versus low-exposure (<0.3) gap is
53.4 points. Mean observed valid-tissue intensity was 0.496 in v3 and 0.185
in v4; among v4 sections with observed mean 0.03–0.10, capture was 21/93
(22.6%), versus 39/50 (78.0%) above 0.30. Raw/exact-black/imperfect-brush
v3 capture was 79.5/88.9/89.7%; v4 was 56.6/32.5/55.9%. These subgroups
were not plane matched, especially the small black/brush groups, and cannot
rank acquisition modes causally.

The predeclared ≥20-point cohort and exposure-gradient criteria passed, and
the subject-bootstrap interval is positive. The result locates a serious
dark-section *proposal* problem before anatomical candidate scoring: even a
truth-chosen branch in the v4 blind beam is often not near the observed tissue.
It does not establish an inability to infer arbitrary planes in principle,
nor final fitted-map or physical-animal accuracy. DeepSlice also reports
degraded performance at extreme contrast and recommends making anatomical
landmarks discernible, but that coronal, often multi-section and externally
pretrained method is not a performance proxy for our single arbitrary-plane
model ([Carey et al., 2023](https://www.nature.com/articles/s41467-023-41645-4)).

**Decision:** target direct pose/encoder robustness to photometric variability
on independently drawn sections, retaining bright and acquired real-donor
performance. Evaluate blind proposal capture *before* returning to the joint
fit-feedback selector. Do not scale the failed 130 score, presume that more
background variants solve this, use public/final animals, calibrate electrode
probabilities, or install the current model in the GUI.

Frozen output: `I:/AnatomyTracker/runs/v3_v4_pose_capture_131b`.
Completion receipt SHA-256:
`6b26a239cf8b48db89a2e8783fa75ec6792440d3fb83e8598b05664325e17577`.
Config, summary, rows and draws SHA-256:
`8c672922ecf4ca62dff42405d256295f8cfc0beb9fbc491d018fa542bd7f14ae`,
`8e52dcdba7d353d225935cfedc2da56862ad6b17a948ded26e23d7fdc6f31ee3`,
`11c00aa8bf01dfcf6b9527723ba3464b78eeb634a49ac0f41128fddc4d39e29f`,
and `10ac56bb2aa354ce6cd2ed2c013feee2d6d9bde8103a0d45f879afb7a23c9000`.
