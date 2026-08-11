# DeepSlice mouse ONNX models

These are self-contained ONNX opset-18 conversions of the official DeepSlice 1.2.8 mouse primary and secondary networks. They preserve the original Xception and dense-layer weights and are loaded only after SHA-256 verification by the tracker.

| Model | ONNX SHA-256 |
| --- | --- |
| primary | `90ce8d4662f53a602035a99d5145c0e6ae8924cde7f9de440cf6b74f79c791ac` |
| secondary | `2d7b5e44d9dc4aa6009df6c3cc7e8a0cbb9fd33dc63a8bd2ac43ea5999237978` |

Validation against official TensorFlow 2.21 inference on the 35-image GLTa dataset gave maximum ensemble differences of `0.00018310546875` in raw OUV output, `0.002515 um` in derived AP, `0.00001483 deg` in L-R tilt, and `0.00002052 deg` in D-V tilt. ONNX Runtime 1.24.4 assigned all 108 graph nodes to DirectML during the GPU validation run.

DeepSlice is by Carey et al., *Nature Communications* 14, 5884 (2023). The upstream license is reproduced in `licenses/DeepSlice-LICENSE.txt`.
