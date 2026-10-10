"""TRAIN-only unpaired check of actual v4/v6 deformation targets and sections."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = os.environ['MPLCONFIGDIR'] = str(root / 'cache')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v6 import sample_one_shot_slide_artifacts_v6
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

out = root / 'runs/v6_local_strain_qc_136'
assert not out.exists()
torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
axis = torch.arange(256, device='cuda') / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(1, -1, 2)
rows, selected = [], {}
seen = set()
arms = (('v4', sample_one_shot_slide_artifacts_v4, 202610101361, 202610101362),
        ('v6', sample_one_shot_slide_artifacts_v6, 202610101363, 202610101364))
target_angles = {'raw': 5., 'exact_black': 25., 'imperfect_brush': 45.}

with torch.inference_mode():
    for arm, sampler, subject_seed, draw_seed in arms:
        rng = np.random.default_rng(subject_seed)
        accepted = 0
        while accepted < 64:
            subjects = rng.integers(len(context['subjects']), size=8).tolist()
            sample = sampler(context, subjects, draw_seed, side=256)
            rigid = rigid_points_090(sample['state'], sample['reflection'],
                                     chart.expand(len(subjects), -1, -1))
            observed = sample['centre'].reshape(len(subjects), -1, 3)
            distance = (rigid - observed).norm(dim=-1).reshape(len(subjects), 256, 256) / 1000
            for i, record in enumerate(sample['provenance']):
                section_id = record['physical_section_id']
                assert section_id not in seen and record['split'] == 'train'
                seen.add(section_id)
                if not bool(sample['eligible'][i]) or accepted == 64:
                    continue
                valid = sample['valid_mask'][i]
                raw_black = (record['mode'] == 'raw' and
                    record['one_shot_slide_artifacts_v3']['parameters']['raw_exterior_code'] == 0)
                if raw_black:
                    assert (sample['inputs'][i, 0][sample['visible'][i] <= .05] == 0).all()
                accepted += 1
                normal = np.abs(record['virtual_unit_normal'])
                angle = float(np.degrees(np.arccos(normal.max())))
                row = {'arm': arm, 'draw_seed': draw_seed, 'draw_row': i,
                    'physical_section_id': section_id,
                    'animal_id': record['base_lineage']['animal_id'],
                    'specimen_id': record['base_lineage']['specimen_id'],
                    'experiment_id': record['base_lineage']['experiment_id'],
                    'mode': record['mode'], 'nearest_cardinal_angle_deg': angle,
                    'valid_pixels': int(valid.sum()),
                    'mean_true_local_displacement_mm': float(distance[i][valid].mean()),
                    'p90_true_local_displacement_mm': float(distance[i][valid].quantile(.9)),
                    'v6_strain': record.get('one_shot_slide_artifacts_v6', {}).get('localized_strain')}
                rows.append(row)
                mode = ('exact_black' if raw_black else record['mode'])
                if arm == 'v6' and mode in target_angles and row['v6_strain']['active']:
                    rank = abs(angle - target_angles[mode])
                    if mode not in selected or rank < selected[mode]['rank']:
                        selected[mode] = {'rank': rank, 'row': row,
                            'state': sample['state'][i:i+1].clone(),
                            'reflection': int(sample['reflection'][i]),
                            'offsets': sample['offsets'][i:i+1].clone(),
                            'weights': sample['weights'][i:i+1].clone(),
                            'image': sample['inputs'][i, 0].cpu().numpy()}
            draw_seed += 1
        print(f'{arm}: {accepted} eligible distinct sections', flush=True)

assert len(rows) == 128 and len(selected) == 3
figure, axes = plt.subplots(3, 2, figsize=(6, 8), facecolor='white')
axis = torch.arange(192, device='cuda') / 192
y, x = torch.meshgrid(axis, axis, indexing='ij')
with torch.inference_mode():
    for j, mode in enumerate(target_angles):
        item = selected[mode]
        centre, frame, basis = full_frame_state_to_components(item['state'])
        chart = torch.stack(((191 / 192 - x) if item['reflection'] else x, y), -1)
        plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        slab = plane[:, None] + item['offsets'][:, :, None, None, None] * frame[:, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(context['atlas'], slab,
            (0., 0., 0.), (25., 25., 25.), item['weights'])[0]
        atlas = (rendered[0] / rendered[1].clamp_min(1e-4)).cpu().numpy()
        atlas[rendered[1].cpu().numpy() < .05] = 0
        image = item['image']
        axes[j, 0].imshow(atlas, cmap='gray', vmin=0, vmax=1)
        axes[j, 1].imshow(image, cmap='gray', vmin=0,
                          vmax=max(.001, float(np.quantile(image, .995))))
        axes[j, 0].set_ylabel(f'{mode}\n{item["row"]["nearest_cardinal_angle_deg"]:.0f}° to cardinal', fontsize=9)
        for panel in axes[j]:
            panel.set_xticks([])
            panel.set_yticks([])
axes[0, 0].set_title('true atlas plane')
axes[0, 1].set_title('generated input; display-stretched')
figure.tight_layout()
out.mkdir(parents=True, exist_ok=False)
figure.savefig(out / 'atlas_input_grid.png', dpi=150)
plt.close(figure)

summary = {'scope': 'TRAIN synthetic only; unpaired v4/v6 independent full-angle planes; not physical validation',
    'per_arm': {arm: {'sections': 64,
        'mean_true_local_displacement_mm': float(np.mean([r['mean_true_local_displacement_mm']
            for r in rows if r['arm'] == arm])),
        'p90_section_mean_mm': float(np.quantile([r['mean_true_local_displacement_mm']
            for r in rows if r['arm'] == arm], .9)),
        'section_means_over_0p5mm': sum(r['mean_true_local_displacement_mm'] > .5
            for r in rows if r['arm'] == arm)} for arm in ('v4', 'v6')},
    'v6_active_strain_sections': sum(r['v6_strain']['active'] for r in rows if r['arm'] == 'v6'),
    'limitation': 'Wider synthetic displacement alone does not establish realistic tissue mechanics or improve model accuracy.'}
(out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
(out / 'grid_selection.json').write_text(json.dumps([selected[m]['row'] for m in target_angles],
    indent=2) + '\n', encoding='utf-8')
sources = ('qc_v6_local_strain.py', 'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_streaming_synthetic_v7.py')
receipt = {'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    for name in sources}, 'output_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
    for name in ('rows.jsonl', 'summary.json', 'grid_selection.json', 'atlas_input_grid.png')},
    'distinct_attempted_physical_sections': len(seen), 'eligible_sections': len(rows)}
(out / 'completed.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary), flush=True)
