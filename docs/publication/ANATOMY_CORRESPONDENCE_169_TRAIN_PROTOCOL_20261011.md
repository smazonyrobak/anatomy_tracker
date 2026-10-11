# 169: anatomy-specific fitter stage (TRAIN only)

The 128, 161 and 162 results showed that learned candidate scores could retain
atlas-support/position shortcuts and that the previous fitted loss did not
improve direct pose prediction. Diagnostic 167 found only 0.018 mm mean local
pose improvement from a measured-fit gradient step. This stage changes the
fitting signal before exposing the direct pose head to it.

The frozen scratch-lineage 132 step-2000 model supplies the direct pose and
local tissue-map components. A randomly initialized matcher is added to that
same model. It encodes the observed image channel and atlas intensity into
normalized local descriptors; absolute atlas position, prior mass, support,
brush availability and candidate state cannot enter either learned descriptor
or the fitted score. A fixed masked average uses support only to exclude atlas
exterior. The correction head sees only local source–atlas correlations plus
normalized raster coordinates, predicts a bounded full-frame residual, and
the atlas is re-rendered at the corrected state. The score is the explicit
post-correction descriptor agreement minus a small correction-effort cost.

Five finite-thickness renders at normal offsets −500, −250, 0, +250 and +500
µm accommodate a bounded amount of real through-plane distortion. Nonzero
depth costs score; this bank is not permission to absorb a globally wrong
plane. Synthetic observed-pixel→CCF truth trains correspondence labels at
valid native pixel sites. Cells crossing an invalid/fragment boundary are
excluded. The truth map and mask never enter inference. A second training
signal compares the same image and fitted plane with its correct atlas
intensity versus histogram-matched wrong-plane intensity pasted only inside
their common tissue interior; position, support and exterior are fixed.

The first run is 6,000 nonblank optimizer updates, at most 7,000 independent
section attempts, on 60 TRAIN deformation bases. Eighty percent of attempts
use support-intersecting 166 geometry; 20% use full-box 151 geometry to retain
sparse/tangential and blank outcomes. Appearance and physical geometry are
sampled independently for every attempt; no deliberate repeated section is
made with a changed background. Ninety-six fixed TRAIN-inner attempts use the
remaining four TRAIN deformation bases. These synthetic bases share one atlas
and are **not** independent biological animals. Checkpoints and raw case rows
are kept at 0, 1,500, 3,000, 4,500 and 6,000 updates. No DEV, public benchmark,
calibration or GUI promotion occurs in this stage.

The main readouts are valid dense-label count, same-support correct-intensity
versus wrong-intensity win rate, true-plane versus genuinely displaced-plane
win rate, and rigid correction error before/after. A useful signal should
improve the first two anatomical decisions on TRAIN-inner cases while reducing
near-plane correction error. We will inspect raw cases and image/atlas ablations
before allowing this fitted signal to train the direct pose distribution in a
matched full-feedback versus detached comparison. Synthetic success alone does
not establish real histology transfer or calibrated probabilities; weak-real
development data and independent animal-level references remain necessary.

The single-case GPU preflight produced 115 valid correspondence labels,
13 fixed-support counterfactual sites and a finite nonzero gradient. A further
14 nonblank draws gave positive-label counts 13–723; 12/14 had at least 12
counterfactual interior sites. These are implementation checks, not accuracy
evidence.
