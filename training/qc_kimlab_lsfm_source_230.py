"""Frozen source-only KimLab LSFM/CCF coordinate check; no model inputs."""

import hashlib
import itertools
import json
import os
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np


root = Path(r"I:\AnatomyTracker\data\kimlab_devccf_v001_source_230")
out = Path(r"I:\AnatomyTracker\reports\kimlab_lsfm_source_230_001")
out.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(out / "mpl")

import matplotlib.pyplot as plt


names = (
    "CCFv3_average_template_ASL_Oriented_u16_25um.nii.gz",
    "KimLabDevCCFv001_iDiscoLSFM2CCF_avgTemplate_ASL_Oriented_25um.nii.gz",
)
expected = (
    (96231610, "bb0b935dd60a567e9744eb4b99ff6300c7693e022408f7b34911ad4835b09ae8"),
    (61578216, "41b4606354a2d5b05a88899e707f9b8690c472b7e46e594646b847af7d4d5b4f"),
)
paths = [root / name for name in names]
for path, (size, digest) in zip(paths, expected):
    assert path.stat().st_size == size
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest

images = [nib.load(str(path)) for path in paths]
assert images[0].shape == images[1].shape == (527, 319, 455)
assert np.array_equal(images[0].affine, images[1].affine)
ccf, lsfm = [np.asarray(image.dataobj) for image in images]
allen_path = Path(r"I:\AnatomyTracker\data\Allen Brain Atlas 25um\average_template_25.nrrd")
annotation_path = Path(r"I:\AnatomyTracker\data\Allen Brain Atlas 25um\annotation_25.nrrd")
allen = nrrd.read(str(allen_path), index_order="F")[0][:527, :319, :455][::4, ::4, ::4].astype(np.float32)
support = nrrd.read(str(annotation_path), index_order="F")[0][:527, :319, :455][::4, ::4, ::4] > 0
ccf_small = ccf[::4, ::4, ::4].astype(np.float32)
lsfm_small = lsfm[::4, ::4, ::4].astype(np.float32)
flips = {"".join(map(str, bits)): float(np.corrcoef(allen.ravel(), ccf_small[tuple(slice(None, None, -1 if bit else 1) for bit in bits)].ravel())[0, 1]) for bits in itertools.product((0, 1), repeat=3)}
lsfm_foreground = lsfm_small > 1000
dice = 2 * np.count_nonzero(support & lsfm_foreground) / (np.count_nonzero(support) + np.count_nonzero(lsfm_foreground))

fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
for row, (label, slices) in enumerate((
    ("AP index 260", (ccf[260], lsfm[260])),
    ("DV index 160", (ccf[:, 160, :], lsfm[:, 160, :])),
    ("ML index 227", (ccf[:, :, 227], lsfm[:, :, 227])),
)):
    for col, section in enumerate(slices):
        lo, hi = np.quantile(section, (0.01, 0.995))
        axes[row, col].imshow(section, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        axes[row, col].set_title(("Allen CCF companion" if col == 0 else "KimLab LSFM") + "; " + label)
        axes[row, col].axis("off")
figure = out / "fixed_orthogonal_views.png"
fig.savefig(figure, dpi=120)
plt.close(fig)

result = {
    "schema": "anatomy-tracker.kimlab-lsfm-source-preflight/230",
    "source_files": [{"name": p.name, "bytes": s, "sha256": h} for p, (s, h) in zip(paths, expected)],
    "source_record": "https://data.mendeley.com/datasets/2svx788ddf/1",
    "allen_template_sha256": hashlib.sha256(allen_path.read_bytes()).hexdigest(),
    "allen_annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "protocol_sha256": hashlib.sha256(Path("docs/publication/KIMLAB_LSFM_SOURCE_PREFLIGHT_230_PROTOCOL_20261011.md").read_bytes()).hexdigest(),
    "shape": list(images[0].shape),
    "dtype": [str(image.get_data_dtype()) for image in images],
    "zooms_mm": [list(map(float, image.header.get_zooms()[:3])) for image in images],
    "affine_ras_mm": images[0].affine.tolist(),
    "qform_code": [int(image.header["qform_code"]) for image in images],
    "sform_code": [int(image.header["sform_code"]) for image in images],
    "finite_fraction": [float(np.isfinite(volume).mean()) for volume in (ccf, lsfm)],
    "allen_companion_correlation_by_index_flips_stride4": flips,
    "allen_companion_direct_mean_absolute_intensity_difference_stride4": float(np.mean(np.abs(allen - ccf_small))),
    "allen_annotation_vs_lsfm_above_1000_dice_stride4": float(dice),
    "lsfm_ccf_within_allen_support_pearson_stride4": float(np.corrcoef(lsfm_small[support], ccf_small[support])[0, 1]),
    "lsfm_quantiles_stride4": np.quantile(lsfm_small, (0, .01, .1, .25, .5, .75, .9, .99, 1)).tolist(),
    "figure_sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
    "label_eligible": False,
    "model_training_performed": False,
}
(out / "source_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({key: result[key] for key in ("shape", "affine_ras_mm", "allen_companion_correlation_by_index_flips_stride4", "allen_annotation_vs_lsfm_above_1000_dice_stride4", "lsfm_ccf_within_allen_support_pearson_stride4")}, indent=2))
