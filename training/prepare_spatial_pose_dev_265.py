"""Freeze fresh physical DEV sections from the existing eight held-out 248 plans."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7 import SUPPORT_SEED, VARIANTS, VIRTUAL_SEED
from training.arbitrary_plane_subject_sampling_v6 import subject_support_bounds_um
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
CACHE = ROOT / 'data/allen_v6_float32_cache_248_001'
GEOMETRY = ROOT / 'data/joint_v7_training_data_001/train_acquisition_geometry.jsonl'
OUT = ROOT / 'data/spatial_pose_fresh_dev_265_001'
DOC = Path('docs/publication/SPATIAL_POSE_UPDATER_265_PROTOCOL_20261012.md')
SEED, SIDE, PER_PLAN = 2026101265, 192, 6
torch.set_num_threads(4)

panel_done = json.loads((PANEL / 'completed.json').read_text())
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
old_records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
prior_ids = {row['provenance']['physical_section_id'] for row in old_records}
geometry = [json.loads(line) for line in GEOMETRY.open(encoding='utf8')]
donors = sorted({row['animal_id'] for row in geometry})
donor_rows = [np.array([i for i, row in enumerate(geometry) if row['animal_id'] == donor]) for donor in donors]
affine = np.array([row['model_pixel_to_ap_dv_ml_um'] for row in geometry])
u, v = affine[:, :, 0] * 96, affine[:, :, 1] * 96
lu = np.linalg.norm(u, axis=1)
u /= lu[:, None]
along = (u * v).sum(1)
v -= along[:, None] * u
lv = np.linalg.norm(v, axis=1)
v /= lv[:, None]
atlas_centre = np.array((528, 320, 456)) * 25. / 2
image_centre = affine[:, :, 2] + 48 * (affine[:, :, 0] + affine[:, :, 1])
nuisance = np.column_stack(((u * (image_centre - atlas_centre)).sum(1),
    (v * (image_centre - atlas_centre)).sum(1), lu, lv, along / lv))
atlas_np = np.load(CACHE / 'atlas_float32.npy', mmap_mode='r', allow_pickle=False)
selected = np.random.default_rng(SUPPORT_SEED).choice(np.flatnonzero(atlas_np[1] > 0), 4096, replace=False)
support_ccf = (np.array(np.unravel_index(selected, atlas_np.shape[1:])).T.astype(np.float64) + .5) * 25
atlas = torch.from_numpy(np.array(atlas_np)).cuda()
del atlas_np

OUT.mkdir(parents=True, exist_ok=False)
records = []
for ordinal, index in enumerate(range(2000, 2008)):
    directory = PANEL / f'subject_{index:08d}'
    subject = json.loads((directory / 'subject.json').read_text())
    plan = _read_raw_artifact(directory, subject['plan_files'])
    with np.load(directory / 'inverse_map.npz', allow_pickle=False) as arrays:
        displacement = torch.from_numpy(np.moveaxis(arrays['displacement_um'], -1, 0).copy()).cuda()
        lower = arrays['lower_unscaled_um'].copy()
        upper = arrays['upper_unscaled_um'].copy()
        centre = arrays['global_centre_um'].copy()
        scale = arrays['global_scale'].copy()
    keys = ('accepted_coarse_coefficients_um', 'coarse_origin_um', 'coarse_spacing_um',
            'accepted_fine_coefficients_um', 'fine_origin_um', 'fine_spacing_um',
            'global_scale', 'frozen_center_um')
    parameters = [torch.as_tensor(plan['state'][key], device='cuda', dtype=torch.float64) for key in keys]
    forward_support = map_accepted_subject_points_torch_v6(
        torch.as_tensor(support_ccf, device='cuda', dtype=torch.float64), *parameters,
        inverse=False, steps=int(plan['resolved_config']['flow']['steps']), batch_size=1024).cpu().numpy()
    anchor = np.array(subject['mapped_atlas_centre_um'])
    base = {'lineage': subject['lineage'], 'displacement': displacement,
        'lower': torch.as_tensor(lower, device='cuda', dtype=torch.float32),
        'upper': torch.as_tensor(upper, device='cuda', dtype=torch.float32),
        'map_centre': torch.as_tensor(centre, device='cuda', dtype=torch.float32),
        'map_scale': torch.as_tensor(scale, device='cuda', dtype=torch.float32),
        'bounds': subject_support_bounds_um(plan), 'anchor': anchor,
        'support_points': forward_support}
    variants = []
    for variant in range(VARIANTS):
        vrng = np.random.default_rng(np.random.SeedSequence([VIRTUAL_SEED, index, variant]))
        virtual_scale = np.exp(vrng.uniform(-.1, .1, 3))
        variants.append({'virtual_index': index * VARIANTS + variant,
            'base_index': 0, 'variant': variant,
            'virtual_subject_id': f"{subject['lineage']['subject_id']}-virtual-affine-v7-{variant:03d}",
            'base_lineage': subject['lineage'], 'anchor_um': anchor.tolist(),
            'scale_ap_dv_ml': virtual_scale.tolist(),
            'bounds_um': (anchor + virtual_scale * (base['bounds'] - anchor)).tolist()})
    context = {'atlas': atlas, 'bases': [base], 'subjects': variants,
               'geometry': geometry, 'donor_rows': donor_rows, 'nuisance': nuisance}
    for local_index in range(PER_PLAN):
        section = 1000 + local_index
        selector = np.random.default_rng(np.random.SeedSequence([SEED, index, section, 4]))
        variant = int(selector.integers(VARIANTS))
        draw_seed = SEED * 100000 + ordinal * 1000 + section
        sample = sample_one_shot_slide_artifacts_v7(context, [variant], draw_seed, side=SIDE)
        source = sample['provenance'][0]
        source['split'] = 'development'
        source['subject_plan_receipt_sha256'] = plan['receipt_sha256']
        source['inverse_map_sha256'] = subject['inverse_map_sha256']
        identity = source['physical_section_id']
        assert identity not in prior_ids
        filename = f'subject_{index:08d}_fresh_{section:04d}.npz'
        np.savez_compressed(OUT / filename, inputs=sample['inputs'][0].cpu().numpy(),
            target_state=sample['state'][0].cpu().numpy(),
            target_centre_um=sample['centre'][0].cpu().numpy(),
            valid_mask=sample['valid_mask'][0].cpu().numpy(),
            reflection=sample['reflection'][0].cpu().numpy())
        records.append({'file': filename,
            'sha256': hashlib.sha256((OUT / filename).read_bytes()).hexdigest(),
            'animal_id': subject['lineage']['animal_id'],
            'specimen_id': subject['lineage']['specimen_id'],
            'experiment_id': subject['lineage']['experiment_id'],
            'section_id': f"{subject['lineage']['subject_id']}-section-{section:08d}",
            'appearance_mode': source['mode'], 'eligible': bool(sample['eligible'][0]),
            'valid_pixels': int(sample['valid_mask'][0].sum()),
            'provenance': source})
        print(json.dumps({'plan': ordinal + 1, 'section': section,
                          'eligible': records[-1]['eligible']}), flush=True)
    del plan, parameters, forward_support, base, context, displacement

(OUT / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'physical_sections': len(records), 'eligible': sum(row['eligible'] for row in records),
    'synthetic_plans': 8, 'prior_physical_ids_disjoint':
        not ({row['provenance']['physical_section_id'] for row in records} & prior_ids),
    'parent_panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'atlas_cache_completed_sha256': hashlib.sha256((CACHE / 'completed.json').read_bytes()).hexdigest(),
    'geometry_sha256': hashlib.sha256(GEOMETRY.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'records_sha256': hashlib.sha256((OUT / 'records.jsonl').read_bytes()).hexdigest(),
    'scope': 'new physical DEV draws on existing eight synthetic plans, not biological validation',
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'fresh_dev_frozen', 'sections': len(records)}), flush=True)
