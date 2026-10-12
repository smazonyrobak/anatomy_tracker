"""Matched clean-atlas, observed-image, and tissue-mask plane-score diagnostic."""
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.run_smooth_image_fit_295 import image_objective, rigid_render


SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/CLEAN_ATLAS_SIGNAL_298_PROTOCOL_20261012.md'
PARENT = ROOT / 'reports/blind_candidate_smooth_fit_296_001'
RUN = ROOT / 'reports/clean_atlas_signal_298_001'
SIDE = 192
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


parent_receipt = json.loads((PARENT / 'completed.json').read_text())
parent_config = json.loads((PARENT / 'config.json').read_text())
assert parent_receipt['candidate'] == parent_config['candidate'] == 296
assert sha256(PARENT / 'inner_panel.pt') == parent_config['inner_panel_sha256']
assert sha256(PARENT / 'sections.jsonl') == parent_receipt['output_sha256']['sections.jsonl']
panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
sections = [json.loads(line) for line in (PARENT / 'sections.jsonl').open()]
assert len(panel) == len(sections) == 32
assert [item['physical_section_id'] for item in panel] == [row['physical_section_id'] for row in sections]
context = load_streaming_synthetic_v7_64('cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']

config = {'candidate': 298, 'parent': 296, 'no_training_or_model_update': True,
    'panel_count': 32, 'candidate_count': 16,
    'arms': ['artifact_common', 'clean_common', 'artifact_privileged_sites',
             'clean_privileged_sites'],
    'protocol_sha256': sha256(PROTOCOL),
    'source_sha256': {name: sha256(SOURCE / name) for name in
        ('clean_atlas_signal_298.py', 'run_smooth_image_fit_295.py',
         'arbitrary_plane_full_frame_primitives.py',
         'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'parent_sha256': {name: sha256(PARENT / name) for name in
        ('completed.json', 'config.json', 'inner_panel.pt', 'sections.jsonl')},
    'atlas_provenance': context['provenance'],
    'score': 'unchanged 296 zero-logit image objective; no fitting',
    'privileged_mask_role': 'saved 512 truly valid surviving tissue sites; diagnostic only',
    'clean_image_role': 'unranked true atlas-plane finite-thickness rendering; diagnostic only',
    'dev_cal_final_or_public_benchmark_used': False,
    'calibrated': False, 'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

axis = (torch.arange(SIDE, device='cuda') + .5) * (2 / SIDE) - 1
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
base_grid = torch.stack((xx, yy), -1)[None]
logits = torch.zeros((1, 2, 16, 16), device='cuda')
rows = []
with (RUN / 'scores.jsonl').open('w') as output:
    with torch.inference_mode():
        for item, section in zip(panel, sections):
            states = item['candidate_states'].cuda()
            reflections = item['candidate_reflections'].cuda()
            offsets = item['offsets'].cuda()
            weights = item['weights'].cuda()
            candidate_render, _, _, _ = rigid_render(context['atlas'], states,
                reflections, offsets.expand(16, -1), weights.expand(16, -1))
            true_render, _, _, _ = rigid_render(context['atlas'],
                item['true_state'].cuda(), item['true_reflection'].cuda(), offsets, weights)
            true_support = true_render[:, 1:2]
            clean = torch.where(true_support > .05,
                true_render[:, :1] / true_support.clamp_min(1e-4), 0)
            images = {'artifact': item['observed_image'].cuda(), 'clean': clean}
            site_mask = torch.zeros(SIDE * SIDE, device='cuda')
            site_mask[item['site_indices'].cuda()] = 1
            masks = {'common': item['common_mask'].cuda().float()[None],
                     'privileged_sites': site_mask.reshape(1, SIDE, SIDE)}
            scores = {f'{image}_{mask}': [] for image in images for mask in masks}
            for candidate in range(16):
                state = states[candidate:candidate + 1]
                reflection = reflections[candidate:candidate + 1]
                basis = full_frame_state_to_components(state)[2]
                rendered = candidate_render[candidate:candidate + 1]
                for image_name, image in images.items():
                    for mask_name, mask in masks.items():
                        value, _ = image_objective(logits, image, rendered, mask,
                            reflection, basis, offsets, base_grid)
                        scores[f'{image_name}_{mask_name}'].append(float(value))
            for candidate, value in enumerate(scores['artifact_common']):
                assert abs(value - section['candidates'][candidate]['zero_image_objective']) < 1e-5
            true_basis = full_frame_state_to_components(item['true_state'].cuda())[2]
            truth_scores = {}
            for image_name, image in images.items():
                for mask_name, mask in masks.items():
                    value, _ = image_objective(logits, image, true_render, mask,
                        item['true_reflection'].cuda(), true_basis, offsets, base_grid)
                    truth_scores[f'{image_name}_{mask_name}'] = float(value)
            pair = section['near_far_pair']
            for arm, values in scores.items():
                winner = int(np.argmin(values))
                row = {'physical_section_id': item['physical_section_id'],
                    'base_index': item['base_index'], 'arm': arm,
                    'obliquity_deg': item['obliquity_deg'],
                    'brush_available': item['brush_available'],
                    'mask_pixel_count': int(masks[arm.split('_', 1)[1]].sum()),
                    'candidate_scores': values,
                    'winner_index': winner,
                    'winner_rigid_site_error_mm':
                        section['candidates'][winner]['rigid_site_error_mm'],
                    'true_control_score': truth_scores[arm],
                    'near_far_eligible': pair['eligible'],
                    'near_rank': pair['near_rank'], 'far_rank': pair['far_rank'],
                    'near_preferred': values[pair['near_rank']] < values[pair['far_rank']]
                        if pair['eligible'] else None}
                output.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
            print(json.dumps({'event': 'section', 'completed': len(rows) // 4,
                'of': len(panel)}), flush=True) if len(rows) % 32 == 0 else None

summary = {}
for arm in config['arms']:
    selected = [row for row in rows if row['arm'] == arm]
    eligible = [row for row in selected if row['near_far_eligible']]
    summary[arm] = {'n': len(selected),
        'mean_winner_rigid_site_error_mm': float(np.mean([
            row['winner_rigid_site_error_mm'] for row in selected])),
        'near_far_eligible': len(eligible),
        'near_far_wins': sum(row['near_preferred'] for row in eligible),
        'mean_true_control_score': float(np.mean([
            row['true_control_score'] for row in selected]))}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
receipt = {'candidate': 298, 'parent_completed_sha256': config['parent_sha256']['completed.json'],
    'rows': len(rows), 'summary': summary,
    'output_sha256': {name: sha256(RUN / name) for name in
        ('config.json', 'scores.jsonl', 'summary.json')},
    'no_training_or_model_update': True, 'calibrated': False,
    'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
