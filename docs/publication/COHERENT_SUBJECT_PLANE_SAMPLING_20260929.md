# Arbitrary-plane coverage for coherent subjects

`training/arbitrary_plane_subject_sampling_v6.py` supplies a small NumPy sampler
for the subject-coordinate adapter. It is not wired into the active local002
experiment or evidence of learned accuracy.

The accepted subject map is `Phi(p) = c + scale * (Exp(v)(p) - c)`.
Its existing componentwise global velocity bound `b` implies a componentwise
unit-time displacement bound `b`. This also holds for the implemented fixed-step
RK4 integrator: its positive stage weights sum to the elapsed time. Thus the
mapped full-CCF box is enclosed by

`[c + scale*(lower-b-c), c + scale*(upper+b-c)]`.

The strictly positive scale multiplies both the source extent and the displacement
allowance. The code uses the accepted coefficient bound, not the sampled audit
grid's maximum displacement. The caller authenticates the frozen plan once at
its generation/load boundary; the sampler does not repeat that expensive work.

Normals are uniform on the projective sphere (Gaussian direction followed by
deterministic antipodal canonicalization), with independent uniform in-plane
roll. Conditional on a normal, signed centre-plane offset is uniform over the
conservative box-projection interval extended by the supplied finite PSF support:
`[-radius-max(z), radius-min(z)]`. This includes every centre plane intersecting
the mapped brain, as well as slab-only intersections and empty planes. The
distribution is not claimed to be uniform over tissue area or biological cuts.
Empty/marginal draws must be retained and labelled by downstream observability,
never silently redrawn. Plane coverage is continuous in principle, not a claim
that a finite training sample visits every orientation/offset.

Each canvas covers the entire bounding box projected into its two in-plane axes.
OUV uses the existing `x/W, y/H` raster convention; its spans include `W/(W-1)`
and `H/(H-1)` so the first/last sampled pixel centres bracket those projections.
There is no finite-FOV tissue rejection. Partial crops, section-specific scale
and acquisition damage are separate later augmentations, not assumptions baked
into this sampler. Random plane draws within a subject share its anatomy but
are not yet an ordered coherent serial-section acquisition.

The caller owns/preserves the NumPy RNG state and section identities. The return
includes exact OUV, normals, rolls, offset intervals, subject bounds, plan and
realization IDs. All sections/augmentations from one subject belong to the same
split. These synthetic anatomies still originate from one real Allen atlas;
synthetic-subject separation cannot establish biological-animal generalization.
