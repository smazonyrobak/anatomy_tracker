"""Optional Allen Nissl intensity for independent arbitrary-plane TRAIN draws."""

import hashlib
from pathlib import Path

import nrrd
import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen


NISSL_PATH = Path('I:/AnatomyTracker/data/Allen Brain Atlas 25um/ara_nissl_25.nrrd')
NISSL_SHA256 = '16596c2c03935b7ff58f1fed41bac2aa21423f39d2c17d3fc400387f054b3bdf'
NISSL_BYTES = 136_232_577
NISSL_Q01, NISSL_Q99 = 337.62844116, 23347.95179688


def bind_nissl_appearance_224(context):
    with NISSL_PATH.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert NISSL_PATH.stat().st_size == NISSL_BYTES and digest == NISSL_SHA256
    nissl, header = nrrd.read(str(NISSL_PATH), index_order='F')
    assert allen._header_binding(header) == {**allen._EXPECTED_COMMON_HEADER_V6, 'type': 'float'}
    support = context['atlas'][1].cpu().numpy() > 0
    assert nissl.shape == support.shape and np.isfinite(nissl).all()
    quantiles = np.quantile(nissl[support], (.01, .99))
    assert np.allclose(quantiles, (NISSL_Q01, NISSL_Q99), atol=.01)
    intensity = np.ascontiguousarray(np.clip((nissl - NISSL_Q01) /
        (NISSL_Q99 - NISSL_Q01), 0, 1), dtype=np.float32)
    intensity[~support] = 0
    atlas = torch.empty_like(context['atlas'])
    atlas[0] = torch.from_numpy(intensity).to(atlas.device)
    atlas[1] = context['atlas'][1]
    binding = {'intensity_path': str(NISSL_PATH), 'intensity_sha256': digest,
        'intensity_raw_bytes': NISSL_BYTES, 'index_order': 'F',
        'normalization': {'in_annotation_support_q01': NISSL_Q01,
                          'in_annotation_support_q99': NISSL_Q99, 'clip': [0, 1]},
        'support_path': str(allen.ANNOTATION_PATH_V6),
        'support_sha256': allen.ANNOTATION_RAW_SHA256_V6,
        'geometry': 'Allen CCFv3 2017, AP/DV/ML 25 um, same as v6',
        'physical_targets': 'unchanged inverse-deformation map; Nissl is image intensity only'}
    return {**context, 'nissl_atlas_224': atlas, 'nissl_binding_224': binding}


def select_atlas_appearance_224(context, seed):
    nissl = bool(np.random.default_rng(np.random.SeedSequence([int(seed), 224])).integers(2))
    if nissl:
        provenance = {**context['provenance'], 'atlas_binding': context['nissl_binding_224']}
        return {**context, 'atlas': context['nissl_atlas_224'], 'provenance': provenance}, 'nissl'
    return context, 'average_template'
