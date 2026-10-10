# 131b: independent confirmation of the v4 pose-capture gap

Freeze the 128 treatment step-6,000 model and its exact blind-16 beam and
observed-valid-pixel rigid-gauge metric. Draw four new eligible physical
sections from each of the 64 independent local-deformation TRAIN subjects in
each appearance cohort: 256 v3 and 256 v4. Use fresh disjoint seed prefixes,
fresh virtual-affine variants and fully independent plane/artifact draws; do
not deliberately pair or duplicate physical sections across backgrounds.
Preserve all failed draws, IDs, complete provenance and source hashes.

Report best-beam and direct-top-one mean error and capture within 1.5 mm,
including exposure, observed-intensity, mode and nearest-cardinal-angle
strata. Bootstrap the 64 subject bases (not individual sections) for a 95%
interval on the v3-minus-v4 beam-capture difference; compare high-exposure
v4 (≥0.6) with low-exposure v4 (<0.3), but note their planes are not matched.
Treat a ≥20-percentage-point v3 advantage and a ≥20-point high-versus-low
v4 exposure gradient, with a positive subject-bootstrap interval, as strong
evidence of a brightness-sensitive capture failure. This still does not prove
causality or real-animal transfer. No training, fitting-score promotion,
calibration, GUI integration or public benchmarking follows automatically.

The completed 131 source is recoverable from commit `4728d6a`. This protocol
uses the same script with a new source hash and output `v3_v4_pose_capture_131b`;
the first output remains frozen unchanged.
