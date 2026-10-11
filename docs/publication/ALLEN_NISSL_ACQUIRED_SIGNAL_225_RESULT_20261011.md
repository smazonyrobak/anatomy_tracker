# 225 TRAIN-only Allen Nissl acquired-signal result

The preregistered comparison in `ALLEN_NISSL_ACQUIRED_SIGNAL_225_PROTOCOL_20261011.md` failed its promotion gate. Keep the average Allen template as the default appearance source; do not start a large Nissl-based training run on the assumption that Nissl repairs acquired-image matching. The pinned Nissl source remains available for a future explicitly matched experiment.

The completed result is `I:\AnatomyTracker\runs\allen_nissl_acquired_signal_225_train_001_retry1` (98 TRAIN sections, 784 candidate pairs, 679 eligible pairs). Each fraction below is donor-equal: fraction of eligible wrong-plane pairs for which the recorded weak-affine plane scored better. The compared planes, finite-thickness renderer, annotation support and pair eligibility were identical across appearance arms.

| TRAIN cohort and candidate type | Eligible pairs | Donors | Average template | Nissl | Permuted Nissl |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coronal, all | 399 | 58 | 0.9697 | 0.6335 | 0.4387 |
| Coronal, normal offset | 173 | 55 | 0.9939 | 0.7030 | 0.4652 |
| Coronal, tilt | 226 | 58 | 0.9612 | 0.6034 | 0.4368 |
| Sagittal, all | 280 | 40 | 0.6223 | 0.6667 | 0.4665 |
| Sagittal, normal offset | 120 | 40 | 0.7375 | 0.7125 | 0.4479 |
| Sagittal, tilt | 160 | 40 | 0.5313 | 0.6313 | 0.4813 |

Nissl's sagittal tilt gain is 0.1000, but its coronal all-pair loss is 0.3362; the preregistered maximum tolerated loss was 0.05. This alone decisively fails the gate. The Nissl permutation control is worse than intact Nissl, so the sagittal effect is not merely support geometry, but it is not a general acquired-image remedy. These are weak-affine, mostly near-cardinal sections, not verified physical ground truth for steep obliques or full tissue deformation.

An independent read-only audit replayed all 98 section identities, eligibility decisions, donor-equal aggregates, and the predeclared gate from raw `pairs.jsonl`. The average-template arm reproduces the frozen 180 results on the same donor/eligibility set (maximum score difference 1.19e-7). Output SHA-256: `config.json` `42556ed1ed72c799ced07a22d66515436a3f3c0f247520784cb8af4095c5b83b`; `pairs.jsonl` `e93381ac603cb246f3e4fc3bf26ae4c9e50991995d5ae3e0a60964f3c7909a3e`; `summary.json` `526a7dc47033e035148b54a92b09a0f23a962bf20355e94c3a5b4e5d241ed28d`. The first invocation's partial output directory is not a result and remains untouched; the protocol records its serialization error and the unchanged-rules retry.

This comparison changes no model checkpoint, inference path, calibration claim, or GUI integration. The next development bottleneck is still blind arbitrary-plane selection: retain candidate anatomy evidence through pose selection and train the coupled pose/local-fit components, with explicit checks that anatomy rather than support or positional shortcuts drives the decision. No DEV/CAL/FINAL animal split or public benchmark was consumed here.
