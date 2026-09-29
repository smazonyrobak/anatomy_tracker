# One actual-section CPU mapping comparison

The opt-in FP64 Torch inverse mapper is numerically consistent with the frozen
NumPy targets for the selected actual section. This is **not GPU validation**,
a forward-map check, model performance, or a biological-generalization result.
The NumPy generator remains the default.

Run `I:/AnatomyTracker/runs/joint_v6_subject_torch_cpu_check_001` exited 0.
Selection was the first horizontally reflected section whose three observation
modes were support-eligible: section 2 of `coherent_subject_sections_001`.
All 9,216 canonical-centre and 82,944 observed S9-slab queries were retained,
including exterior points. Canonical queries were reconstructed from the saved
OUV using the original x/W,y/H arithmetic; slab queries were loaded directly.
The saved CCF targets were not regenerated or modified.

| Physical-coordinate error (µm) | Canonical | Observed slab | All 92,160 points |
|---|---:|---:|---:|
| Maximum absolute component | 1.8190e-12 | 3.6380e-12 | 3.6380e-12 |
| Component RMS | 4.8230e-14 | 4.9255e-14 | 4.9153e-14 |
| Maximum point L2 | 1.8331e-12 | 3.6380e-12 | 3.6380e-12 |
| Point L2 RMS | 8.3536e-14 | 8.5312e-14 | 8.5136e-14 |

All outputs were finite and passed the predeclared maximum-component tolerance
of 1e-8 µm. Fixed 8-step RK4, 8,192-point tiles, FP64 CPU, four intra-op threads
and one inter-op thread were used. Mapping took 13.4645 s; recorded preparation
plus mapping/output time was 13.6954 s. These are single-run timings, not an
acceleration benchmark: the earlier section-generation timing also included
rendering, annotations, appearances and serialization. No GPU was used and the
active rehearsal output tree was not accessed.

## Exact provenance

Implementation commit: `e9d1c63a96864ee320d92dca7d985587c7d81dc0`.
Torch mapper source SHA256:
`741429afe91f2b40e294df86f31789bcd4601742774ed23bf2919b0752d88793`.
The actual frozen plan, section metadata/arrays and source bindings were hashed
and checked; the plan content receipt was recomputed without sampler replay.
Plan ID: `379a1cddefaede38495ee0910e911426159551d6f4b00adad55cd8f1b1e0dc18`.
Plan receipt: `496669aacf08f9a5cee3aedc4fcf3a600cfe9ced106fd1358c3d4bef2cf8c7bc`.
The complete animal/specimen/experiment/section IDs, source hashes, original
reference-source commit and input file hashes are retained in `completed.json`.
Runtime was Python 3.11.15, NumPy 2.4.4 and Torch 2.11.0+cu128 on CPU.

- `completed.json` SHA256: `d47c588b0aac806b00e819bfd969eae967aa8afe8771ed38a6f8801b5497daad`.
- `protocol.json` SHA256: `b1dfd9ba557965a60c2fbc4d37aebd5fadc450bf1e00ed8d71f685cf20c5bc1e`.
- `mapped_coordinates_and_errors.npz` SHA256: `ba364eda4d925837a9c956559e9492b45418411117e67be29f4e2565c754dbe2`.
- Flat comparison-script SHA256: `b1fe6305fcf37c74b182ff6fce7ea1fa220f41e50857ebf2cd9ef85ac4b05458`;
  retained as `comparison_source.py` in the run directory, with original at
  `I:/AnatomyTracker/tmp/check_joint_v6_subject_torch_cpu_001.py`.

This one real-query comparison supports the port's inverse-map arithmetic, not
universal boundary equivalence. A bounded actual-plan GPU comparison is still
required before GPU target preparation; frozen target identities must not be
silently replaced by the new evaluator.
