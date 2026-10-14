"""Gross 25-um grid/anatomy check for six donor-distinct Allen red volumes."""
import hashlib
import json
import os
from itertools import product
from pathlib import Path
from zipfile import ZipFile

os.environ['MPLCONFIGDIR'] = 'I:/AnatomyTracker/cache/matplotlib'

import matplotlib.pyplot as plt
import nrrd
import numpy as np

from training.arbitrary_plane_allen_atlas_binding_v6 import ANNOTATION_PATH_V6, TEMPLATE_PATH_V6

root = Path('I:/AnatomyTracker')
sources = ((156198187, 607436902, 112942577, 'TRAIN'),
           (168615344, 607628055, 160252067, 'TRAIN'),
           (307742524, 315602749, 287634601, 'TRAIN'),
           (257636467, 607916488, 182849231, 'DEV'),
           (526501457, 527105701, 517337553, 'DEV'),
           (649361916, 650149968, 642454692, 'DEV'))
template, _ = nrrd.read(str(TEMPLATE_PATH_V6), index_order='F')
annotation, _ = nrrd.read(str(ANNOTATION_PATH_V6), index_order='F')
brain = annotation != 0
small_brain = brain[::4, ::4, ::4]
small_template = np.log1p(template[::4, ::4, ::4].astype(np.float32))
fig, axes = plt.subplots(1 + len(sources), 3, figsize=(12, 17))
rows = []

for row, (experiment, file_id, donor, split) in enumerate(((0, 0, 0, 'atlas'), *sources)):
    if row == 0:
        volume = template
    else:
        archive = root / 'data/allen_stpt_red_310_001' / f'{experiment}_{file_id}.zip'
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        with ZipFile(archive) as source:
            header = source.read('resampled_red.mhd').decode()
            assert 'DimSize = 528 320 456' in header and 'ElementSpacing = 25 25 25' in header
            assert 'Offset = 0 0 0' in header and 'ElementType = MET_USHORT' in header
            volume = np.frombuffer(source.read('resampled_red.raw'), dtype='<u2').reshape(
                (456, 320, 528)).transpose(2, 1, 0)
        sampled = np.log1p(volume[::4, ::4, ::4].astype(np.float32))
        correlations = {}
        for flips in product((False, True), repeat=3):
            candidate = sampled
            for axis, flip in enumerate(flips):
                if flip:
                    candidate = np.flip(candidate, axis)
            correlations[str(tuple(int(flip) for flip in flips))] = float(
                np.corrcoef(candidate[small_brain], small_template[small_brain])[0, 1])
        rows.append({'experiment_id': experiment, 'well_known_file_id': file_id,
                     'donor_id': donor, 'split': split, 'archive_sha256': digest,
                     'inside_q01_q99': np.quantile(volume[brain], (.01, .99)).tolist(),
                     'inside_outside_median': [float(np.median(volume[brain])),
                                               float(np.median(volume[~brain]))],
                     'flip_correlations': correlations})
    for col, index in enumerate((264, 160, 228)):
        section = (volume[index] if col == 0 else
                   volume[:, index] if col == 1 else volume[:, :, index])
        support = (brain[index] if col == 0 else
                   brain[:, index] if col == 1 else brain[:, :, index])
        axes[row, col].imshow(section, cmap='gray', vmin=0,
                              vmax=np.percentile(section[support], 99.5))
        axes[row, col].set_axis_off()
        axes[row, col].set_title(f"{('atlas' if row == 0 else split + ' donor ' + str(donor))} / {('AP', 'DV', 'ML')[col]}")

fig.tight_layout()
fig.savefig(root / 'reports/allen_stpt_red_six_donor_qc_310.png', dpi=130)
plt.close(fig)
(root / 'reports/allen_stpt_red_six_donor_qc_310.json').write_text(
    json.dumps(rows, indent=2) + '\n', encoding='utf8')
print([(row['donor_id'], row['flip_correlations']['(0, 0, 0)'],
        max(row['flip_correlations'], key=row['flip_correlations'].get)) for row in rows])
