# 175 TRAIN-only local atlas-fit direction gate

The 174 full-pose corrector failed its atlas-intensity control. Before another
model run, test whether a simple explicit image fit has a useful *direction*
from the frozen 132 model's actual blind proposals. Use 128 new independent
TRAIN sections, alternating v3/v4 appearance, random base and physical plane,
and the exact acquisition PSF. No development, final animal, public benchmark,
trained matcher, smart-brush mask, or synthetic tissue-validity mask enters
the fit. The synthetic truth is used only after blind proposals to choose
diagnostic candidates and define a direction toward the true plane.

For each eligible section inspect (a) the model's highest-prior blind branch
and (b) the physically closest blind branch with the correct reflection and
0.5--3 mm five-site error. In each, compare the rendered plane at the original
pose and at plus/minus 20% of the exact local full-frame residual toward
synthetic truth. All three image scores use exactly the same fixed mask: the
7-pixel-eroded *original candidate* atlas support, at least 100 pixels. The mask is computable
without truth, tissue segmentation, or knowledge of the test direction;
record how much support it retains after both moves. Score is a smooth
absolute-Pearson surrogate `sqrt(r²+1e-4)` after 3-pixel box smoothing of
both observed and atlas images. Repeat after cyclically permuting intensity
*among pixels inside that mask* by one-third of the mask length; the same
fixed mask and tissue intensity values are retained, without importing
out-of-support zeros. This is an anatomy-mismatch control, not a zero-atlas
control. Measure the sign of the exact autograd
directional derivative at the original pose as well as the finite move; if
the renderer gives no gradient, this fit cannot train the pose head.

The gate to justify a differentiable fit-loss training run is at least 65%
positive-direction agreement among physically monotonic moves in the
0.5--3 mm near-candidate stratum on both appearance versions, at least 15
percentage points over the permuted-intensity control, and at least 60%
positive autograd direction there. These are
screening thresholds, not a claim of success on unseen animals or a tuned
benchmark. Report counts, ineligible reflections/support, physical move,
support retention, angle strata, accepted rows and all attempted-draw
provenance regardless of outcome. Do not scale an energy that
fails this local test; keep arbitrary-plane support while looking for a more
identifiable visual signal.
