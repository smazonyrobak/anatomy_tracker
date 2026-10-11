"""TRAIN-only KimLab LSFM intensity on the unchanged Allen physical atlas."""

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen


LSFM_PATH = Path('I:/AnatomyTracker/data/kimlab_devccf_v001_source_230/KimLabDevCCFv001_iDiscoLSFM2CCF_avgTemplate_ASL_Oriented_25um.nii.gz')
LSFM_BYTES = 61_578_216
LSFM_SHA256 = '41b4606354a2d5b05a88899e707f9b8690c472b7e46e594646b847af7d4d5b4f'
LSFM_Q01, LSFM_Q99 = 48.0, 26428.0


def bind_lsfm_appearance_232(context):
    with LSFM_PATH.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert LSFM_PATH.stat().st_size == LSFM_BYTES and digest == LSFM_SHA256
    image = nib.load(str(LSFM_PATH))
    assert image.shape == (527, 319, 455)
    assert np.allclose(image.affine, [[0, 0, .025, 0], [-.025, 0, 0, 0],
                                      [0, -.025, 0, 0], [0, 0, 0, 1]], atol=1e-8)
    support = context['atlas'][1].cpu().numpy() > 0
    assert support.shape == allen.ATLAS_SHAPE_AP_DV_ML_V6
    assert not (support[-1].any() or support[:, -1].any() or support[:, :, -1].any())
    lsfm = np.pad(np.asarray(image.dataobj), ((0, 1), (0, 1), (0, 1)), mode='constant')
    assert lsfm.shape == support.shape and np.isfinite(lsfm).all()
    q01, q99 = map(float, np.quantile(lsfm[support], (.01, .99)))
    assert (q01, q99) == (LSFM_Q01, LSFM_Q99)
    intensity = lsfm.astype(np.float32)
    intensity -= np.float32(q01)
    intensity /= np.float32(q99 - q01)
    np.clip(intensity, 0, 1, out=intensity)
    intensity[~support] = 0
    intensity = np.ascontiguousarray(intensity)
    atlas = torch.empty_like(context['atlas'])
    atlas[0] = torch.from_numpy(intensity).to(atlas.device)
    atlas[1] = context['atlas'][1]
    binding = {'source_contrast': 'KimLab P56 registered LSFM average',
        'intensity_path': str(LSFM_PATH), 'intensity_sha256': digest,
        'intensity_raw_bytes': LSFM_BYTES, 'source_shape_ap_dv_ml': [527, 319, 455],
        'zero_pad_high_index_ap_dv_ml': [1, 1, 1],
        'normalized_intensity_sha256': hashlib.sha256(memoryview(intensity).cast('B')).hexdigest(),
        'normalization': {'in_annotation_support_q01': q01,
                          'in_annotation_support_q99': q99, 'clip': [0, 1]},
        'support_path': str(allen.ANNOTATION_PATH_V6),
        'support_sha256': allen.ANNOTATION_RAW_SHA256_V6,
        'geometry': 'Allen CCFv3 2017, AP/DV/ML 25 um; source affine directly index-aligned',
        'physical_targets': 'unchanged TRAIN inverse-deformation map; LSFM is image intensity only',
        'source_record': 'https://data.mendeley.com/datasets/2svx788ddf/1',
        'source_license': 'CC BY 4.0; retain source attribution and separate Allen-layer rights'}
    return {**context, 'lsfm_atlas_232': atlas, 'lsfm_binding_232': binding}


def select_atlas_appearance_232(context, seed):
    lsfm = bool(np.random.default_rng(np.random.SeedSequence([int(seed), 232])).integers(2))
    if lsfm:
        provenance = {**context['provenance'], 'atlas_binding': context['lsfm_binding_232']}
        return {**context, 'atlas': context['lsfm_atlas_232'], 'provenance': provenance,
                'appearance_source': 'KimLab P56 registered LSFM average'}, 'KimLab LSFM'
    return context, 'Allen average'
