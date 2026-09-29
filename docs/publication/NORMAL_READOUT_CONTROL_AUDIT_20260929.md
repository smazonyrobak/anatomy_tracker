# Direct-normal readout control: failed, but not a dead context layer

Frozen 006 completed 4,000 applied updates with no AMP skips and failed its
predeclared improvement gate. Independent CPU recomputation confirms worse
development capture than 005 at the same step. **Do not extend this control**
or promote it to joint training, deployment or public benchmarking.

| Frozen 4,000-step endpoint |005 base head |006 direct-normal tilt |
| --- | ---: | ---: |
| Joint-cell NLL |9.75639 |10.62410 |
| Normal-marginal NLL |5.62423 |5.84594 |
| Joint-MAP normal error |49.6611° |51.6038° |
| Normal-marginal MAP error |48.4800° |51.0594° |
| Normal mass within 10° |2.8866% |2.1464% |
| Nearest-cell top-32 recall |1.7188% |0.3125% |
| Nearest-cell top-128 recall |5.1563% |1.2500% |

These are absolute antipodal-normal endpoints on 640 sections / 40
single-atlas organizational groups, not biological-animal generalization.
Raw probabilities are uncalibrated. All frozen logits are finite; maximum
absolute log-normalization errors are 8.72e-7 and 7.30e-7 respectively.

## Bounded activation diagnosis

CPU FP32 inference used exactly eight existing support-eligible training rows:
`[0,708,1417,2141,2854,3597,4350,5119]`. Both checkpoints were loaded strictly
into their declared architectures. Only image→encoder→context→query/readout
was executed; no atlas rendering, catalogue inference, training, or GPU work.

| Across-input centered RMS on the same eight rows |005 |006 |
| --- | ---: | ---: |
| Input image |0.1954 |0.1954 |
| Encoder output |0.5028 |0.0519 |
| Pre-GELU context |13.8472 |1.3620 |
| Post-GELU context |8.8697 |0.4480 |
| Normal query |6.2727 |0.4828 |
| Direct-normal logits |not present |0.5314 |

The 006 endpoint does **not** support total dead-GELU collapse on these rows:
no row has all-negative preactivations or an all-near-zero context. Only 1.17%
of its post-GELU entries have magnitude below 1e-8; only 0.78% have an analytical
GELU derivative magnitude below 1e-8. The corresponding 005 fractions are 76.37% / 75.20%,
despite its stronger input-dependent active features. Negative GELU values
alone must not be mistaken for dead units.

006 has substantially smaller activations and input sensitivity. Its final
encoder GroupNorm scale norm is 5.78 versus 8.94 in 005, and all 64 final encoder
normalization biases are negative. These observations do not establish why
optimization diverged, nor whether smaller amplitudes cause poor capture.
The endpoint audit does not exclude a transient earlier plateau/collapse.

Optimization did occur: 006 direct-readout weights moved from zero to L2=17.74;
context weights changed by L2=27.73. Recorded pre-clipping gradient norms are
finite and nonzero, with medians 1.356 versus 9.013 for 005. Both trace and optimizer
state report 4,000 applied updates. Last-100 training joint NLL is 10.4999 versus
9.6280, so 006 is no longer exactly uniform at the endpoint, but remains inferior.
This is a negative result for this fixed optimizer/objective/head control, not
proof that all direct normal readouts are unsuitable.

## Reproduction

Script:
[`training/diagnose_joint_v6_normal_readout_collapse.py`](../../training/diagnose_joint_v6_normal_readout_collapse.py).
Full statistics, parameter deltas and checkpoint receipts:
[`results/normal_readout_collapse_diagnostic_005_006.json`](results/normal_readout_collapse_diagnostic_005_006.json).
Original JSON and fixed-row activations remain under
`I:/AnatomyTracker/runs/joint_v6_normal_readout_collapse_audit_005_006/`.
Original JSON SHA-256:
`2b42008fdf9ad0fb8c016f067a39f6704883d418798da7213fffe62099225000`.
The repository copy changes only text line endings/final newline and has SHA-256
`fc3915a30cf76eaff45f7d9b4ff1d52a9d574494c422887b3c4f675ca577daef`.

006 final checkpoint SHA-256:
`1a0d29a3c7effe1bf67adc92e63b5416e2607eeb6aeaa3d29b5ab1457b362128`.
006 frozen raw development prediction SHA-256:
`b88503c17952b52ad63e630864576f2cfbe08dcf93bce9c32a37e344b52ff5e5`.
