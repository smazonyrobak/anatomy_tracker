# 262: blind-grid geometric coverage before a spatial atlas updater

The 260 direct dense model predicts nearly one constant plane normal on the held-out
full-sphere synthetic panel. Before training another model, test whether a renderer
candidate grid built only from its image-predicted centre and physical spans can
put a true plane within a bounded correction region. This is a **truth-measured
geometry upper bound**, not blind candidate selection, image matching, or a model
accuracy claim.

Use the frozen 260 batch-3,000 checkpoint and exactly the 227 eligible physical
sections in frozen panel 248. Neither section IDs nor appearances are regenerated.
For each section, ignore 260's collapsed normal but retain its predicted centre
and the two singular values of its physical in-plane basis. The inference-time
grid uses the project's fixed 64 antipodal normal anchors, candidate centres at
that predicted centre plus -2, 0, or +2 mm along each anchor normal, and each
predicted span multiplied by either 1 or 1.8. It would use eight in-plane roll
bins and two raster reflections; those cover any roll to within 22.5 degrees,
so this preflight measures the remaining factors without rendering images.

For each saved true plane, mark the grid *geometrically correctable* only if at
least one normal/offset/scale candidate simultaneously has: unsigned normal
error at most 15 degrees, physical centre error at most 3 mm, and both sorted
physical span log-ratios at most 0.7 in absolute value. These bounds correspond
to a proposed local updater; true labels are used only to score coverage, never
to create or rank an inference-time candidate. This test is optimistic because
it does not measure shear, atlas support, visible-tissue ambiguity, or whether
an image network can select or learn the correction.

Report the case and equal-plan correctable fraction, component failures, centre
and span residual distributions, and appearance/visibility strata. A follow-on
spatial image–render model is justified only if at least 70% of cases and at
least six of eight synthetic plans each have at least 60% correctable cases.
If this gate fails, widen or repair **framing** before training the head; do not
mistake a missing true-plane candidate for a ranking failure. No acquired,
calibration or final animals, public benchmark, GUI route, or probability claim.
