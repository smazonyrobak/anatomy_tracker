# 248 pinned float32 atlas cache

The independent v7 panel has intermittently crashed inside native NumPy operations while re-decoding and normalizing the 25-µm Allen NRRDs in the same process as historical-plan and map work. Freeze the exact, previously pinned Allen v6 float32 two-channel atlas once in a small isolated process on I:. The existing decoder must verify the raw NRRDs, decoded arrays, foreground support, and exact normalized atlas receipt before writing `atlas_float32.npy`. Bind its file hash, source and raw-atlas hashes to a completion receipt. This is a reference-data cache, **not** a trained model, feature or pseudolabel.

The panel and its exploratory 247 readout will hash-check the cache file and receipt, confirm the pinned shape/dtype and recorded array receipt, then load those exact bytes without rerunning NRRD normalization. No plane, artifact, pose or deformation target changes. The cache does not reduce the need for independent animal-level validation.
