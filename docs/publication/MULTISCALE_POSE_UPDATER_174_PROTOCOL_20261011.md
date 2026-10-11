# 174: multiscale atlas-conditioned full-pose correction

This is the first substantial correction curriculum in the existing randomly
initialized 132 lineage, **not** a deployable joint model. The 173 diagnostic
distinguished the exact synthetic plane from wrong planes in 137/143 cases,
but distinguished the better of two model proposals in only 86/143. The new
cell therefore predicts a bounded nine-coordinate full-frame update from the
observed section and a finite-thickness atlas render, rerenders at the updated
plane, and shares its weights across coarse and fine spatial scales. A small
through-plane render bank is part of each step. It does not receive absolute
pose coordinates; the original probabilistic beam and its no-shift action are
retained. The 132 parent is frozen during this stage so the new correction
mechanism can be measured without confounding direct-pose drift.

Training uses fresh independently sampled TRAIN physical planes across all 64
synthetic deformation bases, current v3/v4 acquisition/artifact appearances,
and 132 blind proposals. Synthetic truth can select and supervise hard and
near candidates **during training only**. It must not construct the inference
beam, score a proposal, or enter the correction inputs. Preserve each draw's
source, base, seed and section provenance. The corrected physical pose, not
just a falling network loss, is the outcome. The initial 2,000-update run is
one staged exposure; extend training only if frozen development evidence and
learning curves warrant it. Any “best” checkpoint selected from training
metrics is only a proxy until the development panel is evaluated.

After the runner exits, audit its frozen source, parent, draw, checkpoint and
raw-result hashes. Evaluate unchanged checkpoint candidates on the existing
frozen 132 v4 and v3 development panels, preserving their eight synthetic
deformation plans and separate physical-section IDs. Report blind-16 original
and corrected best-available errors, prior-selected/no-shift error, corrected
selected error, capture within 1.5 mm, exposure and oblique-angle strata, and
the same scores after atlas **intensity** is zeroed while support remains. The
intensity-null arm is a conditional anatomy-use control, not a retrained model.
Report acquired coronal and sagittal DEV donor-equal disagreement against the
inherited weak Allen references separately; those references are not expert
ground truth. No public benchmark or final animal is touched.

A useful advance would close at least 0.30 mm of selected v4 error against the
parent/no-shift result, gain at least 0.20 mm in truth-chosen best-of-16
correction, and show a material anatomy-dependent advantage over the
intensity-null control without a severe v3 or acquired-donor regression.
These are decision aids, not permission to claim a calibrated or shippable
model, and a promising sub-threshold checkpoint remains preserved. If the
correction does not use atlas intensity or cannot improve a near candidate,
do not feed its fit score back to direct pose. If it does, continue the **same
model** with a physically constrained local tissue map and a validated
polarity-invariant fit loss that has an auditable gradient into the pose heads.
The later full-versus-detached feedback comparison must establish that fitting
actually teaches coordinate prediction.
