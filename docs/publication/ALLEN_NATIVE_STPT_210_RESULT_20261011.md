# 210: eight TRAIN native-red coordinate maps

Acquisition 208 froze eight distinct Allen STPT donors before image inspection. The native red stacks contain 140 physically acquired coronal sections per donor at 100 µm section spacing; the downsampled XY grid is approximately 50 µm. Their section images and Allen deformation fields were acquired on `I:`. The completed 210 receipt is `I:/AnatomyTracker/data/allen_stpt_native_maps_210_train_001/completed.json`, SHA-256 `5a555e7afef162c51fe6b071d692a4b95faed5d07bd838634b47a8c04b72d077`.

The maps convert each native red voxel centre to CCFv3 2017 AP/DV/ML micrometres by inverting Allen's authored CCF-to-native field. Donor 14383 reuses completed 209; seven further maps were built independently. All eight have shape `140×210×280×3`, finite coordinates, exact animal/specimen/experiment bindings, and independently checked source/map hashes. Worst-section 95th-percentile *numerical inverse* residuals, in µm:

| TRAIN donor | Experiment | Residual |
|---|---:|---:|
| 14383 | 100140756 | 1.810 |
| 14451 | 100140949 | 5.398 |
| 14453 | 100141214 | 0.774 |
| 14543 | 100141222 | 0.747 |
| 14544 | 100141223 | 2.745 |
| 14598 | 100141473 | 3.376 |
| 14599 | 100141273 | 6.323 |
| 14600 | 100141435 | 2.566 |

Six smallest donors are the predeclared 212 gradient set; 14599 and 14600 are inner held animals. No DEV/CAL/FINAL donor is in this source set. These residuals check numerical inversion of the **same upstream field**, not agreement with independent anatomical landmarks. The Allen registrations remain weak reference labels. Reslicing these coronal stacks at oblique angles is a virtual interpolation experiment, not physically acquired arbitrary-angle validation. The next fixed step is the 212 matched synthetic-versus-native continuation with unchanged parent initialization, followed by the 213 physical-section and 214 retention readouts.
