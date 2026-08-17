# Runtime assets

`python setup_runtime.py` downloads the exact checksum-pinned assets used by the validated desktop installation:

- the DeepSlice 1.2.8 primary and secondary mouse ONNX models;
- the evaluated AtlasPose ONNX model;
- the dense anatomical-registration ONNX model;
- the Allen CCFv3 25 µm template, annotation, structure lookup, and 3-D mesh.

DeepSlice is distributed under the MIT license reproduced in `licenses/DeepSlice-LICENSE.txt`. Allen Institute atlas content and models derived from it are provided for noncommercial research use subject to the [Allen Institute Terms of Use](https://alleninstitute.org/terms-of-use/) and [Citation Policy](https://alleninstitute.org/citation-policy/). Cite Carey et al. (2023) for DeepSlice and Wang et al. (2020) for Allen CCFv3.

The setup script verifies every downloaded file against a pinned SHA-256 digest. Virtual environments, Python packages, generated executables, training workspaces, and caches are not included in the runtime release.
