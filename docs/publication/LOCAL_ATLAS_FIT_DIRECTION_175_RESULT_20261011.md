# 175 frozen TRAIN gate: a narrow, anatomically informative fit direction exists

The predeclared TRAIN-only local-direction gate ran to completion on 128 fresh,
independently drawn physical sections (64 v3, 64 v4; 134 attempts). It used the
frozen 132 blind 16-branch beam, no trained matcher, and a finite-thickness
atlas render. Each branch's fixed eroded scoring mask came solely from its
original atlas support; no synthetic tissue-validity mask or truth entered the
fit. The post-exit hashes of `config.json`, `draws.jsonl`, `rows.jsonl`, and
`summary.json` matched `completed.json`. No DEV image, final animal, expert
real label, or public benchmark was used.

The diagnostic truth chose the physically closest blind branch with matching
reflection and 0.5--3 mm five-site error. For that *oracle-selected candidate
only*, the +20% local full-frame move toward truth increased smooth absolute
image/atlas correlation relative to the -20% move in 31/39 v3 sections (79.5%)
and 41/43 v4 sections (95.3%). The support-preserving permuted-intensity
control did so in 21/39 (53.8%) and 21/43 (48.8%). The exact autograd
directional derivative at the original pose was positive in 29/39 (74.4%)
and 35/43 (81.4%). All 82 eligible moves were physically monotonic in the
five-site error. Mean support retention at the moved poses exceeded 99.9%,
with the minimum above 99.2%, so an edge entering/leaving the mask is not a
plausible explanation for these near-candidate rates. The signal was present
in all observed angle strata, but the most oblique strata contain only six
v3 and five v4 sections and cannot establish robust arbitrary-angle accuracy.

This meets the protocol's local-direction screening thresholds. It does
**not** solve global pose or inference-time branch selection. Such a near
correct-reflection branch existed with at least 100 scoring pixels for only
39/64 v3 and 43/64 v4 images. The highest-prior branch was often too far away
or had the wrong reflection: only 26/64 v3 and 32/64 v4 highest-prior branches
were eligible for this directional test. Among those branches the fit signal
was weaker (17/26 v3, 20/32 v4) and near the permuted-intensity control,
especially for v4. Their mean starting errors were 6.68 and 5.42 mm. The
positive near-candidate finding is therefore conditional on an oracle choice
and must not be presented as deployed or validation accuracy.

**Decision:** pursue differentiable atlas-fit feedback only inside a captured
local basin; it is unjustified as a stand-alone global objective from the
current highest-prior branch. The next model stage must (1) preserve
probabilistic multimodal arbitrary-plane proposals, (2) learn or explicitly
search which *blind* branch has a credible local fit without synthetic truth,
(3) use the locally informative fit gradient for pose correction and route it
to the direct pose head, and (4) constrain tissue deformation so it cannot
make a wrong plane look cheap. A matched full-feedback versus stopped-gradient
comparison, with the same compute and candidate set, is required before
claiming joint optimization. DEV panels remain frozen for that comparison;
no DeepSlice benchmark or calibration claim is appropriate yet.

Frozen raw evidence:

- `I:/AnatomyTracker/runs/local_atlas_fit_direction_175_train_001/completed.json`
- `I:/AnatomyTracker/runs/local_atlas_fit_direction_175_train_001/summary.json`
- `I:/AnatomyTracker/runs/local_atlas_fit_direction_175_train_001/rows.jsonl`
- `I:/AnatomyTracker/runs/local_atlas_fit_direction_175_train_001/draws.jsonl`

`summary.json` SHA-256:
`1fb00a7d1bdc62615a669d8e07fa642a3c3e89f1d03b078429950b1bf4204315`.
