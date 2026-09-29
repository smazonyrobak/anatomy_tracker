# Eight-case memorization control: CPU runtime failure, no learning conclusion

Two bounded CPU launches used the completed003 F64/H128 joint model with its
original smooth proposal head, corrected FP32 computation and explicit
`proposal_normal_readout_count=None`. Initialization was fresh, seed2026092908;
AdamW learning rate0.001, weight decay0.0001, batch4 alternating the same eight
images. No GPU tensors or pretrained weights were used.

Selection was deterministic: joint-pack rows `[2,3,1,5,6,4,8,9]`, unique labels
`[20645,75451,66358,11939,70789,37160,55681,20264]`, all pose/dense weights one,
1,133–4,024 foreground tissue pixels, raw/accurate/imperfect mode counts3/3/2.
Both launches began with identical joint NLL11.495839 and mean normal error
68.271031 degrees. This nearly uniform initialization metric is not a learning
result. Neither process reached the first100-step milestone.

Run001 exited1 without Python traceback. Windows Application Error1000 at
2026-09-29 11:10:00 local reported process12504, native exception0xc0000005 in
`nvdxgdmal64.dll_unloaded`, report `ea394e92-3f5b-44cd-8aa2-02f60baa72a9`.
Approximately50GiB physical RAM was free; initial process RSS was0.89GB and
startup peak2.25GB. Run002 repeated unchanged computation with unbuffered output
and Python faulthandler. It also exited1; the fatal access-violation stack was
inside `torch.nn.Linear.forward`, called by the proposal geometry embedding at
`arbitrary_plane_coarse_proposal_v6.py:196`, during a training forward pass.

Failure records and unchanged experiment/initial-metric artifacts are under
`I:/AnatomyTracker/runs/joint_v6_eight_case_cpu_overfit_001` and `_002`.
No current005 output was opened or changed. This diagnostic currently says
**nothing about eight-case memorization or model correctness**. The next useful
action is the identical fresh control on GPU after005 exits, not further OS
forensics or interpretation of nonexistent training results.
