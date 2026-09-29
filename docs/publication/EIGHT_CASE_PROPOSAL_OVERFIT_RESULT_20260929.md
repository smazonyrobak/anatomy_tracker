# Eight-case control: the unchanged proposal model can memorize the inputs

Fixed endpoint: 1,000 updates, exit0, **8/8 exact full catalogue cells correct**.
Mean joint NLL falls from11.495817 to0.02991259; final normal-marginal NLL is
0.00466360. All8×98,304 final log probabilities are finite; a separate CPU read
of the saved array confirms all eight argmax labels and endpoint NLL.

Mean normal error to the **continuous physical truth** is3.17391059 degrees,
maximum4.46652816 degrees. Error to the corresponding **target catalogue normal**
is zero: every predicted cell equals its target. Thus the remaining continuous
angle is precisely this eight-case catalogue-quantization floor, not failed
normal classification.

The entire sparse recorded trajectory, rather than a selected best endpoint:

| Updates | Joint NLL | Continuous-normal error, degrees | Exact full cells |
| --- | ---: | ---: | ---: |
| 0 | 11.495817 | 68.271031 | 0/8 |
| 100 | 5.651221 | 18.283144 | 3/8 |
| 200 | 2.426344 | 3.173911 | 6/8 |
| 300 | 0.985184 | 12.522597 | 7/8 |
| 400 | 2.595057 | 10.178562 | 6/8 |
| 500 | 0.119851 | 3.173911 | 8/8 |
| 600 | 0.058732 | 3.173911 | 8/8 |
| 700 | 0.026028 | 3.173911 | 8/8 |
| 800 | 0.013100 | 3.173911 | 8/8 |
| 900 | 0.008162 | 3.173911 | 8/8 |
| 1000 | 0.029913 | 3.173911 | 8/8 |

Alternating-batch Adam oscillation is visible at300/400 and the final NLL rises
relative to900. No best-step substitution was made. Full-cell predictions stay
correct at every recorded checkpoint from500 onward.

This uses exactly the eight CPU-control cases and fresh seed2026092908:
F64/H128 completed003 architecture, original smooth geometry head, optional
normal readout **None**, FP32 without AMP, full joint NLL only, AdamW0.001,
weight decay0.0001, gradient clipping5, batch4 alternating two fixed halves.
Input modes are raw/accurate/imperfect3/3/2. No model weights, external features
or pseudolabels were loaded. The two prior native CPU failures remain preserved;
see `EIGHT_CASE_CPU_RUNTIME_FAILURE_20260929.md`.

The GPU run briefly overlapped005 under explicit authorization. Its wall time
was64.98seconds, peak allocated GPU memory471,845,376bytes, endpoint RSS
1,620,828,160bytes and process startup peak2,248,302,592bytes. These are resource
records, **not timing comparisons or performance benchmarks**. No005 output
tree was accessed or altered.

This demonstrates working optimization and sufficient representational capacity
to memorize these eight observations. It does **not** establish generalization,
calibration, correct image-to-anatomy truth in every row, or exclude all labeling
bugs: arbitrary labels can also be memorized. It therefore narrows—but does not
settle—the explanation for43.7-degree development error. No new run follows
from this note.

Run: `I:/AnatomyTracker/runs/joint_v6_eight_case_gpu_overfit_001`.
Driver: `training/run_joint_v6_eight_case_overfit.py`, commit `c356d4f`.
Exact cases, row receipts, source hashes and configuration are in `experiment.json`.

| Artifact | SHA-256 |
| --- | --- |
| `experiment.json` | `4d46c93c471d0c3adfe89351d418f97b498a16c77b3886db635552d0b5c69c11` |
| `metrics.jsonl` | `0fbdb2b109e41ecefb8ab3d906c14f6624a52d0f69e10f2c0740679b5701d51e` |
| `completed.json` | `57450d15778b1c3eb99e082622a17efacd3d4c95a937e531c25703c84a8e3ca3` |
| `final_raw_log_probability.npy` | `5c836639d77a64a7ba24dfd9e4de7c5c8cc299c1bd222b72cbcc475363152af8` |
