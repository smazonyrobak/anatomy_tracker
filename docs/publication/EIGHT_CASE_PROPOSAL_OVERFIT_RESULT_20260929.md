# Eight-case control: the unchanged proposal model can memorize the inputs

Fixed FP32 endpoint: 1,000 updates, exit 0, **8/8 exact full catalogue cells correct**.
Mean joint NLL falls from 11.495817 to 0.02991259; final normal-marginal NLL is
0.00466360. All 8×98,304 final log probabilities are finite; a separate CPU read
of the saved array confirms all eight argmax labels and endpoint NLL.

Mean normal error to the **continuous physical truth** is 3.17391059 degrees,
maximum 4.46652816 degrees. Error to the corresponding **target catalogue normal**
is zero: every predicted cell equals its target. Thus the remaining continuous
angle is precisely this eight-case catalogue-quantization floor, not failed
normal classification.

The entire sparse trajectory, including the subsequent matched encoder-AMP
control, rather than selected best endpoints:

| Updates | FP32 NLL | FP32 normal error, degrees | FP32 exact cells | AMP NLL | AMP exact cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.495817 | 68.271031 | 0/8 | 11.495817 | 0/8 |
| 100 | 5.651221 | 18.283144 | 3/8 | 5.625709 | 3/8 |
| 200 | 2.426344 | 3.173911 | 6/8 | 2.406685 | 6/8 |
| 300 | 0.985184 | 12.522597 | 7/8 | 0.997195 | 7/8 |
| 400 | 2.595057 | 10.178562 | 6/8 | 0.331201 | 8/8 |
| 500 | 0.119851 | 3.173911 | 8/8 | 0.146249 | 8/8 |
| 600 | 0.058732 | 3.173911 | 8/8 | 0.097146 | 8/8 |
| 700 | 0.026028 | 3.173911 | 8/8 | 0.033080 | 8/8 |
| 800 | 0.013100 | 3.173911 | 8/8 | 0.016973 | 8/8 |
| 900 | 0.008162 | 3.173911 | 8/8 | 0.016774 | 8/8 |
| 1000 | 0.029913 | 3.173911 | 8/8 | 0.010472 | 8/8 |

FP32 alternating-batch Adam oscillation is visible at 300/400 and its final NLL
rises relative to 900. No best-step substitution was made. Full-cell predictions
stay correct at every recorded checkpoint from 500 onward.

The paired run keeps the same fresh seed/cases/architecture/optimizer/objective,
adding actual FP16 encoder autocast and GradScaler(initial scale 256), with the
corrected density head remaining FP32. It completes 1,000/1,000 applied updates,
**zero skips**, final scale 256. Endpoint NLL is 0.01047235, normal NLL 0.00035237,
and exact-cell recall 8/8. Its continuous-normal error is 67.135948 degrees
initially, 28.552930 at 100, then 3.17391059 at every 200–1000 checkpoint; its
endpoint target-normal error is zero. The eight-case fit survives actual training
precision; this does not establish that AMP is generally better or exclude all
precision/generalization problems.

Both runs use exactly the eight CPU-control cases and fresh seed 2026092908:
F64/H128 completed003 architecture, original smooth geometry head, optional
normal readout **None**, full joint NLL only, AdamW 0.001, weight decay 0.0001,
gradient clipping 5, batch 4 alternating two fixed halves.
Input modes are raw/accurate/imperfect 3/3/2. No model weights, external features
or pseudolabels were loaded. The two prior native CPU failures remain preserved;
see `EIGHT_CASE_CPU_RUNTIME_FAILURE_20260929.md`.

Both GPU runs briefly overlapped 005 under explicit authorization. FP32 wall
time was 64.98 seconds, peak GPU allocation 471,845,376 bytes, endpoint RSS
1,620,828,160 bytes and startup peak RSS 2,248,302,592 bytes. AMP wall time was
71.20 seconds, peak GPU allocation 466,803,200 bytes and endpoint RSS
1,648,562,176 bytes. These are resource records, **not timing comparisons or
performance benchmarks**. No 005 output tree was accessed or altered.

This demonstrates working optimization and sufficient representational capacity
to memorize these eight observations. It does **not** establish generalization,
calibration, correct image-to-anatomy truth in every row, or exclude all labeling
bugs: arbitrary labels can also be memorized. It therefore narrows—but does not
settle—the explanation for 43.7-degree development error. This paired diagnostic
series is complete; no further runs follow from this note.

FP32 run: `I:/AnatomyTracker/runs/joint_v6_eight_case_gpu_overfit_001`.
AMP run: `I:/AnatomyTracker/runs/eight_case_gpu_amp_overfit_001`.
Driver: `training/run_joint_v6_eight_case_overfit.py`, commits `c356d4f` (FP32)
and `0bf7789` (AMP).
Exact cases, row receipts, source hashes and configuration are in `experiment.json`.

| Artifact | SHA-256 |
| --- | --- |
| `experiment.json` | `4d46c93c471d0c3adfe89351d418f97b498a16c77b3886db635552d0b5c69c11` |
| `metrics.jsonl` | `0fbdb2b109e41ecefb8ab3d906c14f6624a52d0f69e10f2c0740679b5701d51e` |
| `completed.json` | `57450d15778b1c3eb99e082622a17efacd3d4c95a937e531c25703c84a8e3ca3` |
| `final_raw_log_probability.npy` | `5c836639d77a64a7ba24dfd9e4de7c5c8cc299c1bd222b72cbcc475363152af8` |

AMP artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `experiment.json` | `6adf8819a47165f2f8a355936c14c50b58f179e178be16f852a7b24df5b81906` |
| `metrics.jsonl` | `4d4f810aea572b6123407046ba5ada38415b8053788080b1e004c58c1c55d7d5` |
| `completed.json` | `69a5a5b4ecc25c14b1f973c703794d46ddd5d7fd365726ad245d2cd36db281f8` |
| `final_raw_log_probability.npy` | `c21ff21fed4c9527f42a1a6a2a26b8c038950add00edbe26f44e4a2387d54cb0` |
