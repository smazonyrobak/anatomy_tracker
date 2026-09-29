# Coherent TRAIN candidate retrieval: completed, substantial capture gap

Runner97190 from `6a172de` exited0; independent CPU reconstruction75081 also
exited0. Integrity passes. This was an inference-only TRAIN preparation, not an
optimizer run or a model-qualification gate. Whole A22576 and its matching
98,304-cell/two-reflection gallery remain unchanged. The normalized native
comparison was not modified or inspected through its protected artifacts.

All640 physical sections/1,920 observations from the same eight synthetic
subjects are retained:512 base sections plus128 acquired views, each in raw,
exact-black and imperfect-brush modes. There are1,149 eligible and771 censored
observations. These are not new biological animals. No DEV section was opened.
Image/optional-outline/availability alone ranked candidates; geometry and IDs
are separate supervision/provenance. True query PSF was not an inference input.

## Actual image-ranked capture

Values below are equal-synthetic-subject means within each eligible group.
Plane capture means at least one of128 cells within10 degrees and500um signed-
normal-offset error. Frame RMS uses the four finite-pixel corners and actual
predicted reflection; oracle best-frame/R is explicitly a readout, not selection.

| TRAIN group | Eligible n | Top1 plane error, degrees | Capture128 | Predicted-R frame RMS, mm | Oracle best128x2 RMS, mm | Retained coarse mass |
|---|---:|---:|---:|---:|---:|---:|
| Base raw |311|48.3991|.43409|13.2503|4.8175|.08434|
| Base black |311|48.9775|.50787|12.8319|4.6362|.10687|
| Base imperfect brush |300|49.2161|.39650|12.9187|4.8513|.10332|
| Acquisition raw |76|53.2738|.42455|11.7032|3.5404|.08953|
| Acquisition black |76|50.0730|.51811|11.1439|3.6745|.10877|
| Acquisition imperfect brush |75|51.0714|.44519|11.3011|3.7098|.10884|

The earlier simpler-synthetic capture result does **not** transfer to these
coherent observations. Merely expanding local teacher perturbations is not a
sufficient solution: about half the eligible cases have no near physical plane
in this beam, and even truth-selected beam/frame alternatives remain far away.
Around89–92% of coarse mass lies outside the retained beam. Those masses are
uncalibrated; normalizing the beam cannot turn it into a complete posterior.
Keep both reflection alternatives and an explicit omitted-tail/abstention path.

No dense registration accuracy, biological generalization, calibrated confidence
or DeepSlice superiority follows from these fitted-frame TRAIN proxies. Future
coarse/native training must cover the actual coherent acquisition canvas, not
assume the previous fixed12mm synthetic query distribution is adequate.

## Concrete canvas and native-reach mismatch

Source comparison found the same cache/coarse query encoding and scoring, AP/DV/ML
micrometre coordinates, x/W,y/H canvas convention and one spatial reflection.
A small actual-TRAIN pixel inspection found expected float32[3,96,96] channels
in[0,1], raw outline/availability zero and masked boundary/availability binary.
It found no concrete conversion, axis, reflection or half-voxel bug. This does
not exclude every possible implementation error or isolate a causal effect.

The coherent base sampler covers conservative subject-box projections, whereas
the coarse generator/gallery uses12x12mm charts. Eligible base median U/V spans
are17.482/17.581mm. Acquired views instead have median13.047/14.240mm spans and
1.953mm tangent-centre displacement. These differences coexist with the poor
retrieval result; they are not an experimentally isolated explanation of it.

A separate saved-frame calculation exposes a hard native limitation. With
three recurrent updates bounded by±.12 on each log-diagonal, a12mm initial
positive QR diagonal can reach only[8.3721,17.19995]mm. Using norm(U) and
norm(V-proj_U(V)), **245/311 eligible base raw and black sections (78.78%) and
238/300 eligible brush sections (79.33%) exceed that upper bound**. All-base
count is404/512. None of128 acquisition frames exceeds this span-only bound,
so it cannot explain their poor retrieval. Passing this necessary span test
does not imply normal/offset/roll/translation/shear capture or learned accuracy.

Do not change the ongoing matched local comparison: its truth-near initializer
is different. Before native training from retrieved cells, explicitly revise
the iteration count/update parameterization or initial-frame support to cover
the measured TRAIN range, together with broad-canvas retrieval training.
Keep actual images unchanged; silently resizing coordinate truth or using
truth-selected beam starts would conceal rather than solve this mismatch.

Chart-reach result: `I:/AnatomyTracker/runs/joint_v6_coherent_train_chart_reach_001/result.json`,
SHA256 `109047833c73119af92fa6e826893a24b7472bf61907dc6c55b2871f5cddcb6f`.
This used only the frozen TRAIN fitted frames/IDs, without a model, new renderer
or rereading dense targets. Per-section values and exact source/input hashes
are saved with it.

## Independent verification and frozen bindings

NumPy/SciPy reconstruction used all98,304x2 saved gallery scores per query,
without loading a model or rerendering an image. Largest cell/conditional-R/
component log-probability discrepancies were1.15e-5 or less, within the declared
2e-5 FP32 reconstruction envelope; retained-mass discrepancy was1.12e-7.
34 rows had near-tie CPU ordering differences (largest score gap4.30e-6), with
no top128 membership difference. Explicit NumPy pixel-corner geometry agreed
within7.28e-12um and normal error within5.59e-13 degrees; capture decisions,
predicted/oracle reflection and oracle ranks agree exactly. Aggregate summaries
agree within3.64e-12. Checkpoint/gallery/catalogue, saved outputs and source
bindings were checked; underlying raw section files were not redundantly
rehashed by the independent auditor after the runner checked them.

Output: `I:/AnatomyTracker/runs/joint_v6_coherent_train_candidates_001`.
Runner elapsed9.3286s excludes setup/hashing and is not a hardware benchmark.
Auditor elapsed43.55s, CPU only.

- Completion: `5f075e6ce34db1527c0c172757c6995a59fc8d7d42a833fbe25bf59942838ba2`.
- Candidates: `823b4386eff75fa07855f7087a5d5775017cc94b29cc443e1b9c67006f65512d`.
- Supervision sidecar: `967ed49cea03bf924596fa8735fee09bd8ad4fda9c996d030da86c3632225a5e`.
- Independent audit: `995272fc036b7e1011bae56fa01678ef21622b79e8f6e1d28879e5f247e1af6f`.
- Executed independent source: `ef9a41ff5789006296e0aa184e64485c9e29b19a520112465072c1ae22566e4f`.
- Whole checkpoint: `e22ff8e13a09b8c624c64518ca65a0ac9feda2c823dde30b6b8b5707995d52d8`.
- Matching gallery: `5219117c897bf68fb6139b5bd482bfdd05f5c52e2b9a5a7296c9936d0f8d0b36`.

Exact IDs, original full-catalogue/component masses and the FP32 roundoff
normalizer remain saved. Any encoder/gallery-contract change invalidates these
cached descriptors/candidates; they are not feature teachers for future training.
