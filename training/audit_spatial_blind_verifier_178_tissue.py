"""Read-only, retrospective tissue-site audit of the frozen 178 TRAIN panel; not its gate."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
run = root / 'runs/spatial_blind_verifier_178_train_001'
source = Path(__file__).resolve().parent
completed = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert hashlib.sha256((run / 'config.json').read_bytes()).hexdigest() == completed['config_sha256']
for name, digest in completed['output_sha256'].items():
    assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
for name, digest in config['source_sha256'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
assert hashlib.sha256((source.parent / 'docs/publication/SPATIAL_BLIND_VERIFIER_178_PROTOCOL_20261011.md').read_bytes()).hexdigest() == config['protocol_sha256']
parent = root / 'runs/v4_pose_adaptation_132'
assert hashlib.sha256((parent / 'completed.json').read_bytes()).hexdigest() == config['parent_completion_sha256']
assert hashlib.sha256((parent / 'joint_step_02000.pt').read_bytes()).hexdigest() == config['parent_checkpoint_sha256']
assert config['parent_checkpoint_sha256'] == json.loads((parent / 'completed.json').read_text())['checkpoint_sha256']['2000']
best = json.loads((run / 'best_checkpoint.json').read_text())
step = completed['best_step']
assert best['step'] == step
assert hashlib.sha256((run / f'verifier_step_{step:05d}.pt').read_bytes()).hexdigest() == completed['best_checkpoint_sha256'] == best['checkpoint_sha256']

draws = [json.loads(line) for line in (run / 'panel_draws.jsonl').open()]
panel = [json.loads(line) for line in (run / f'panel_{step:05d}.jsonl').open()]
rows = {(row['physical_section_id'], row['arm']): row for row in panel}
assert len(draws) == 128 and len(rows) == 256 and all(row['used'] and row['eligible'] for row in draws)
assert len({row['physical_section_id'] for row in draws}) == 128
assert all(sum(row['base_index'] == base and row['appearance'] == appearance for row in draws) == 8
           for base in config['panel_bases'] for appearance in ('v3', 'v4'))
assert set(config['panel_bases']).isdisjoint(config['gradient_bases'])

context = load_streaming_synthetic_v7_64(device='cpu')
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]]) / 256
results = []
with torch.no_grad():
    for record in draws:
        base, seed, appearance = record['base_index'], record['draw_seed'], record['appearance']
        geometry = sample_panel_geometry_train_166(context, [base], seed, 256)
        sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
                   else sample_one_shot_slide_artifacts_v4)
        sample = sampler(context, [base], seed, 256, source_sample=geometry)
        physical_id = record['physical_section_id']
        assert sample['provenance'][0]['physical_section_id'] == physical_id
        assert int(sample['valid_mask'][0].sum()) == record['valid_pixels']
        ids = record['beam_branch_ids']
        assert len(ids) == len(set(ids)) == 16
        assert record['beam_reflection'] == [branch % 2 for branch in ids]
        states = torch.tensor(record['beam_state12'], dtype=torch.float32)[None]
        reflection = torch.tensor(record['beam_reflection'], dtype=torch.long)[None]
        truth_five = rigid_points_090(sample['state'], sample['reflection'], five)[0]
        error_five = (rigid_points_090(states, reflection, five)[0]
                      - truth_five[None]).norm(dim=-1).mean(-1).div(1000).numpy()
        assert np.allclose(error_five, record['beam_original_five_point_mm'], atol=.01, rtol=1e-4)
        full, null = rows[(physical_id, 'full')], rows[(physical_id, 'null')]
        assert full['direct_branch_id'] == null['direct_branch_id']
        assert full['native_branch_id'] == null['native_branch_id']
        for row in (full, null):
            assert row['beam_branch_ids'] == ids
            assert row['original_five_point_mm16'] == record['beam_original_five_point_mm']
            assert all(row[key] in ids for key in ('direct_branch_id', 'native_branch_id',
                                                  'selected_original_branch_id'))
            assert abs(row['selected_original_mm'] - record['beam_original_five_point_mm'][
                ids.index(row['selected_original_branch_id'])]) < 1e-5
        pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        assert len(pixels) >= 128
        pixels = pixels[torch.linspace(0, len(pixels) - 1, 128).long()]
        sites = torch.stack((pixels.remainder(256), pixels.div(256, rounding_mode='floor')),
                            -1).float() / 256
        truth = rigid_points_090(sample['state'], sample['reflection'], sites)[0]
        error = (rigid_points_090(states, reflection, sites)[0]
                 - truth[None]).norm(dim=-1).mean(-1).div(1000).tolist()
        results.append({'appearance': appearance,
                        'direct': error[ids.index(full['direct_branch_id'])],
                        'native_original': error[ids.index(full['native_branch_id'])],
                        'full_selected_original': error[ids.index(full['selected_original_branch_id'])],
                        'null_selected_original': error[ids.index(null['selected_original_branch_id'])]})

metrics = ('direct', 'native_original', 'full_selected_original', 'null_selected_original')
by_appearance = {}
for appearance in ('v3', 'v4'):
    cases = [row for row in results if row['appearance'] == appearance]
    assert len(cases) == 64
    by_appearance[appearance] = {'sections': len(cases),
        'mean_mm': {key: float(np.mean([row[key] for row in cases])) for key in metrics},
        'capture_le_1p5': {key: float(np.mean([row[key] <= 1.5 for row in cases]))
                           for key in metrics}}
print(json.dumps({'status': 'retrospective_descriptive_non_gate', 'best_step': step,
                  'panel_sections': len(results), 'observed_valid_tissue_sites_per_section': 128,
                  'by_appearance': by_appearance}, allow_nan=False))
