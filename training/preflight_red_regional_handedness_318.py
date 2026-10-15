"""TRAIN-only regional handedness check for registered individual red volumes."""
from zipfile import ZipFile

import nrrd
import numpy as np

from training.arbitrary_plane_allen_atlas_binding_v6 import ANNOTATION_PATH_V6, TEMPLATE_PATH_V6
from training.arbitrary_plane_red_source_mix_311 import ROOT, TRAIN_SOURCES

atlas, _ = nrrd.read(str(TEMPLATE_PATH_V6), index_order='F')
annotation, _ = nrrd.read(str(ANNOTATION_PATH_V6), index_order='F')
atlas = np.log1p(atlas[::4, ::4, ::4].astype(np.float32))
brain = annotation[::4, ::4, ::4] != 0

for experiment, file_id, donor, _ in TRAIN_SOURCES:
    with ZipFile(ROOT / f'{experiment}_{file_id}.zip') as source:
        red = np.frombuffer(source.read('resampled_red.raw'), dtype='<u2').reshape(
            456, 320, 528).transpose(2, 1, 0)[::4, ::4, ::4]
    red = np.log1p(red.astype(np.float32))
    flipped = red[:, :, ::-1]
    scores = []
    for ap in (slice(0, 44), slice(44, 88), slice(88, 132)):
        for ml in (slice(0, 57), slice(57, 114)):
            region = np.s_[ap, :, ml]
            tissue = brain[region]
            truth = atlas[region][tissue]
            direct = np.corrcoef(truth, red[region][tissue])[0, 1]
            mirror = np.corrcoef(truth, flipped[region][tissue])[0, 1]
            scores.append((round(float(direct), 4), round(float(mirror), 4)))
    print(donor, scores, 'direct_wins', sum(a > b for a, b in scores), flush=True)
