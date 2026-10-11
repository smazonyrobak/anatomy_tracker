"""TRAIN-only geometry and appearance audit of two registered individual STPT brains."""

import hashlib
import json
import os
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['XDG_CACHE_HOME'] = os.environ['MPLCONFIGDIR'] = str(root / 'cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


data = root / 'data/allen_stpt_individual_234'
sheet = data / 'MapMySections_EntrantData.xlsx'
assert hashlib.sha256(sheet.read_bytes()).hexdigest() == '21535da73152821026fb57baef33474ef48ce96b9a81c8846c779b58984d696c'
atlas_root = root / 'data/Allen Brain Atlas 25um'
template, _ = nrrd.read(str(atlas_root / 'average_template_25.nrrd'), index_order='F')
annotation, _ = nrrd.read(str(atlas_root / 'annotation_25.nrrd'), index_order='F')
support = annotation[::4, ::4, ::4] > 0
support_points = np.argwhere(annotation[::8, ::8, ::8] > 0) * 8
atlas_coarse = gaussian_filter(template[::4, ::4, ::4].astype(np.float32), 1)
atlas_coarse = (atlas_coarse[support] - atlas_coarse[support].mean()) / atlas_coarse[support].std()
rng = np.random.default_rng(20261011234)
fig, axes = plt.subplots(8, 3, figsize=(10, 23))
rows = []
sources = []
source_vectors = []

for block, (series_id, challenge_id) in enumerate((('1285852977', 'MMS.training.001'),
                                                     ('1283915380', 'MMS.training.002'))):
    path = data / f'{series_id}_resampled_red_25.nii.gz'
    image = nib.load(str(path))
    volume = np.asarray(image.dataobj)
    assert volume.shape == template.shape == annotation.shape == (528, 320, 456)
    q01, q99 = map(float, np.quantile(volume[::4, ::4, ::4][support], (.01, .99)))
    coarse = gaussian_filter(volume[::4, ::4, ::4].astype(np.float32), 1)
    source_vectors.append(coarse[support])
    flips = []
    for bits in range(8):
        flipped = np.flip(coarse, axis=tuple(axis for axis in range(3) if bits & (1 << axis)))
        candidate = flipped[support]
        flips.append(float(np.mean(atlas_coarse * ((candidate - candidate.mean()) / candidate.std()))))
    anatomy = volume[110:420, 60:260:4, 65:390:4].astype(np.float32)
    axis0_adjacent = [float(np.mean(np.abs(anatomy[k:] - anatomy[:-k]))) for k in (1, 2, 3, 4)]
    inplane_100um = [float(np.mean(np.abs(np.diff(anatomy, axis=axis)))) for axis in (1, 2)]
    sources.append({'challenge_id': challenge_id, 'image_series_id': series_id,
        'file': str(path), 'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'shape_ap_dv_ml': list(volume.shape), 'dtype': str(volume.dtype),
        'nifti_affine': image.affine.tolist(), 'nifti_units': image.header.get_xyzt_units(),
        'in_allen_support_q01_q99': [q01, q99],
        'correlation_with_allen_template_by_flip_bits': flips,
        'axis0_mean_absolute_difference_at_25_50_75_100um': axis0_adjacent,
        'inplane_mean_absolute_difference_at_100um_dv_ml': inplane_100um})

    for draw in range(4):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        reference = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(reference, normal)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        angle = rng.uniform(-np.pi, np.pi)
        u, v = np.cos(angle) * u + np.sin(angle) * v, -np.sin(angle) * u + np.cos(angle) * v
        center = support_points[rng.integers(len(support_points))].astype(np.float64)
        projections = (support_points - center) @ np.stack((u, v), axis=1)
        span = 1.08 * np.ptp(projections, axis=0).max()
        midpoint = (projections.max(axis=0) + projections.min(axis=0)) / 2
        xy = np.linspace(-.5, .5, 256, dtype=np.float32)
        xx, yy = np.meshgrid(xy, xy)
        plane = center[:, None, None] + u[:, None, None] * (midpoint[0] + span * xx) \
            + v[:, None, None] * (midpoint[1] + span * yy)
        source_plane = np.zeros((256, 256), np.float32)
        atlas_plane = np.zeros_like(source_plane)
        for shift, weight in ((-2., .12), (-1., .23), (0., .30), (1., .23), (2., .12)):
            points = (plane + shift * normal[:, None, None]).reshape(3, -1)
            source_plane += weight * map_coordinates(volume, points, order=1,
                mode='constant', cval=0, output=np.float32).reshape(256, 256)
            atlas_plane += weight * map_coordinates(template, points, order=1,
                mode='constant', cval=0, output=np.float32).reshape(256, 256)
        source_abs = np.clip((source_plane - q01) / (q99 - q01), 0, 1)
        source_stretch = np.clip(source_plane / max(float(np.quantile(source_plane, .995)), 1), 0, 1)
        atlas_display = np.clip(atlas_plane / 273, 0, 1)
        row = block * 4 + draw
        for col, panel in enumerate((atlas_display, source_abs, source_stretch)):
            axes[row, col].imshow(panel, cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')
        axes[row, 0].text(.02, .98,
            f'{challenge_id}  |  {np.degrees(np.arccos(np.abs(normal).max())):.0f}° from nearest cardinal',
            transform=axes[row, 0].transAxes, va='top', color='white', fontsize=8,
            bbox={'facecolor': 'black', 'alpha': .7, 'pad': 3})
        rows.append({'challenge_id': challenge_id, 'image_series_id': series_id,
            'normal_ap_dv_ml': normal.tolist(), 'center_index_ap_dv_ml': center.tolist(),
            'roll_rad': float(angle), 'slab_offsets_um': [-50, -25, 0, 25, 50],
            'global_normalized_q995': float(np.quantile(source_abs, .995))})
    del volume

for col, label in enumerate(('same Allen CCF plane', 'registered STPT, global absolute scale',
                             'registered STPT, per-plane display stretch')):
    axes[0, col].set_title(label, fontsize=10)
fig.tight_layout()
preview = root / 'previews/allen_stpt_individual_full_sphere_234.png'
fig.savefig(preview, dpi=130)
plt.close(fig)
receipt = {'source_sheet': str(sheet), 'source_sheet_sha256': hashlib.sha256(sheet.read_bytes()).hexdigest(),
    'source_url': 'https://github.com/AllenInstitute/MapMySections/blob/main/MapMySections_EntrantData.xlsx',
    'series': sources, 'same_voxel_source_pair_pearson_in_allen_support':
        float(np.corrcoef(*source_vectors)[0, 1]),
    'rows': rows, 'preview': str(preview),
    'preview_sha256': hashlib.sha256(preview.read_bytes()).hexdigest(),
    'warning': 'CCF-warped channels from coronal STPT, not independent arbitrary-oblique physical sections; red may include reporter fluorescence; model and validation untouched.'}
preview.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({'preview': str(preview), 'series': [s['image_series_id'] for s in sources],
    'flip_correlations': [s['correlation_with_allen_template_by_flip_bits'] for s in sources],
    'q01_q99': [s['in_allen_support_q01_q99'] for s in sources]}, indent=2))
