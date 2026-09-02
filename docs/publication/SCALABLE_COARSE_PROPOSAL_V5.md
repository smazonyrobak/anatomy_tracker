# Scalable coarse proposal v5: engineering premise

This is an internal engineering design, not a scientific result, validation,
benchmark, calibration, or release claim.  It preserves fresh random
initialization and has no prior weights, foundation features, embeddings,
pseudolabels, or automatic-segmentation dependency.

The proposal consumes the existing histology encoding, including the explicit
optional-outline availability channel.  It therefore leaves the established
raw/acquired-background, exact-black-exterior, and imperfect smart-brush
curriculum unchanged and does not require an automatic slice mask.  No data
loader, provenance field, or animal/specimen/experiment identifier is altered.

The production catalogue contains 98,304 physical cells (384 antipodal RP2
normals x 16 support offsets x 16 rolls), each with two exact raster
representations.  Exhaustive finite S=9 retrieval at 48 x 48 therefore
evaluates 2,038,431,744 atlas-volume trilinear sample points and 452,984,832
subsequent 2-D representation-resampling points per input row before feature
encoding.  (The through-plane atlas render is shared by the two raster
representations.)  It then encodes and pairs 196,608 represented atlas images
per row.  The v5 path predicts an uncalibrated
multimodal categorical proposal from the histology encoder and explicit
normal/offset/roll cell geometry, then applies the existing exact finite-S=9
renderer and learned representation marginalization only to top-M=64 cells.
That reduces those scopes to 1,327,104 atlas-volume sample points and 294,912
2-D resampling points, and only 128 represented atlas images, respectively:
exactly 1,536-fold, while preserving unchanged full-resolution top-K recurrent
pose/deformation refinement.

One single-thread CPU engineering microbenchmark (98,304 cells, F=32, eight
mixture components, 16 proposal channels, seven timed inference-only passes)
measured a 117.2 ms median for the 15,016-parameter proposal head, with a
109.9--123.5 ms observed range.  The full probability vector occupies 393,216
bytes and the three transient factor embeddings occupy 18,874,368 bytes in
float32.  This timing is a local scaling measurement only; it is not an
end-to-end runtime or accuracy result.

Complete-catalogue atlas-feature caches are deliberately disabled for v5:
the cascade's exact renderer uses row-specific finite-thickness schedules only
for the proposed cells.  Existing exhaustive checkpoints retain their cache
path unchanged.

The proposal is a mixture of factorized learned energies over canonical RP2
normal, signed normal offset, and canonical in-plane roll frame.  Multiple
components can retain separated modes.  A complete normalized proposal
posterior and omitted mass remain available, but are explicitly uncalibrated.
Exact finite-thickness scores affect top-M selection/reranking only; they are
not misreported as a normalized full-catalogue posterior.  A conditional
cross-entropy trains the exact reranker only on rows whose truth was honestly
retained in top-M.  Training may inject the truth into the final refinement
set only after recording honest proposal and rerank outputs; evaluation rejects
any truth-index input.

The design direction is consistent with global arbitrary single-slice matching
in [SLIV-Reg](https://arxiv.org/abs/2410.18683), the scale and ambiguity exposed
by the [Needles & Haystacks WACV 2025 benchmark](https://openaccess.thecvf.com/content/WACV2025/html/Frolov_Needles__Haystacks_Dataset_and_Benchmark_for_Domain-Agnostic_Image-Based_Rigid_WACV_2025_paper.html),
and the efficiency motivation of [EUReg](https://papers.miccai.org/miccai-2025/0309-Paper1387.html).
The uncertainty outputs remain candidates for later animal-disjoint calibration;
[Hierarchical Uncertainty Estimation for Learning-based Registration in
Neuroimaging](https://openreview.net/pdf?id=w8LMtFY97b) supports using learned
aleatoric uncertainty during transformation fitting and downstream propagation,
but does not establish calibration for this model.

The next scientific gate is a new receipt-bound internal-development run.  It
must measure honest top-M and top-K recall, point accuracy, regional overlap,
failure rate, correction burden, and calibration under strict animal-level
splits before this proposal can replace the exhaustive path in a release
candidate.  The untouched final-test animals and later public/external
benchmarking remain deferred until the method stabilizes.
