# 230: source-only preflight of the KimLab P56 LSFM average

Freeze this source check before downloading or inspecting pixels. The [official KimLabDevCCFv001 record](https://data.mendeley.com/datasets/2svx788ddf/1) describes an adult P56 iDISCO LSFM **average** registered to Allen CCFv3, released on an isotropic 25-µm grid under CC BY 4.0. Isotropic grid spacing does not establish isotropic optical acquisition or independent-animal pose truth.

Download only these two published 25-µm NIfTI files to `I:/AnatomyTracker/data/kimlab_devccf_v001_source_230`; do not write development files to C:. Pin the [official file API](https://data.mendeley.com/api/datasets/2svx788ddf/files?version=1) metadata and verify byte counts and SHA-256 before loading either file:

| File | ID | Bytes | Published SHA-256 |
| --- | --- | ---: | --- |
| `CCFv3_average_template_ASL_Oriented_u16_25um.nii.gz` | `f106cbe6-6fb5-4934-a707-bc5af79701cd` | 96,231,610 | `bb0b935dd60a567e9744eb4b99ff6300c7693e022408f7b34911ad4835b09ae8` |
| `KimLabDevCCFv001_iDiscoLSFM2CCF_avgTemplate_ASL_Oriented_25um.nii.gz` | `028d96bb-3793-424e-8e78-852df2788053` | 61,578,216 | `41b4606354a2d5b05a88899e707f9b8690c472b7e46e594646b847af7d4d5b4f` |

Read NIfTI dimensions, voxel units, qform/sform, affine and finite-intensity fractions. Compare the companion CCF intensity template against our pinned Allen average by an explicit physical-coordinate map, not a guessed `ASL` array flip. Check anatomy on a small, fixed set of orthogonal views, then assess whether LSFM provides distinct brain-internal texture at matched coordinates. Record transformations, metrics, images, and source hashes on I:. If axes cannot be reconciled or anatomical correspondence is poor, reject this source. This is TRAIN-source qualification only: no model training, label promotion, reserved-animal access, uncertainty calibration, GUI change, or public benchmarking. A positive result permits only a small matched source ablation; it does not prove physical arbitrary-plane generalization.
