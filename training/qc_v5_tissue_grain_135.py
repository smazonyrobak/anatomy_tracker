"""Bounded, unpaired TRAIN-only v4/v5 texture and actual-generator image QC."""

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
import torch.nn.functional as F
from scipy import ndimage as ndi

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v5 import sample_one_shot_slide_artifacts_v5
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64


prior = root / 'runs/v3_interior_texture_119_train'
out = root / 'runs/v5_tissue_grain_qc_135'
prior_done = json.loads((prior / 'completed.json').read_text())
prior_summary_bytes = (prior / 'summary.json').read_bytes()
assert hashlib.sha256(prior_summary_bytes).hexdigest() == prior_done['summary_sha256']
prior_summary = json.loads(prior_summary_bytes)
assert prior_done['real_sections'] == 512 and prior_summary['groups']['real']['sections'] == 512
assert not out.exists()

torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
arms = [('v4', sample_one_shot_slide_artifacts_v4, 20261010135, 20261011135),
        ('v5', sample_one_shot_slide_artifacts_v5, 20261012135, 20261013135)]
rows, attempts, selected, seen = [], [], {}, set()
target_angles = {'raw': 5., 'exact_black': 25., 'imperfect_brush': 45.}

with torch.inference_mode():
    for arm, sampler, subject_seed, draw_seed in arms:
        rng = np.random.default_rng(subject_seed)
        accepted = 0
        while accepted < 256:
            subjects = rng.integers(len(context['subjects']), size=8).tolist()
            batch = sampler(context, subjects, draw_seed, side=256)
            image192 = F.interpolate(batch['inputs'][:, :1], (192, 192), mode='area')[:, 0].cpu().numpy()
            valid192 = (F.interpolate(batch['valid_mask'][:, None].float(), (192, 192), mode='area')
                        [:, 0] > .999).cpu().numpy()
            images = batch['inputs'][:, 0]
            exact_black = []
            for i, record in enumerate(batch['provenance']):
                assert record['split'] == 'train'
                assert all(key in record['base_lineage'] for key in ('animal_id', 'specimen_id', 'experiment_id'))
                assert all(key in record for key in ('physical_section_id', 'virtual_subject_id',
                                                     'one_shot_slide_artifacts_v3', 'one_shot_slide_artifacts_v4'))
                assert ('one_shot_tissue_grain_v5' in record) == (arm == 'v5')
                section_id = record['physical_section_id']
                assert section_id not in seen
                seen.add(section_id)
                assert torch.isfinite(batch['state'][i]).all() and torch.isfinite(batch['centre'][i]).all()
                assert (batch['centre'][i][~batch['valid_mask'][i]] == 0).all()
                raw_black = (record['mode'] == 'raw' and
                    record['one_shot_slide_artifacts_v3']['parameters']['raw_exterior_code'] == 0)
                if raw_black:
                    exterior = batch['visible'][i] <= .05
                    assert (images[i][exterior] == 0).all()
                    exact_black.append(int(exterior.sum()))
                used = bool(batch['eligible'][i]) and accepted < 256
                attempts.append({'arm': arm, 'draw_seed': draw_seed, 'row': i,
                    'physical_section_id': section_id, 'eligible': bool(batch['eligible'][i]),
                    'used': used, 'raw_exact_black_exterior_pixels_checked':
                        exact_black[-1] if raw_black else 0})
                if not used:
                    continue
                accepted += 1
                pixels = image192[i]
                foreground = ndi.binary_fill_holes(ndi.binary_closing(pixels > .03, iterations=3))
                labels, count = ndi.label(foreground)
                tissue = labels == np.bincount(labels.ravel())[1:].argmax() + 1 if count else foreground
                interior = ndi.binary_erosion(tissue, iterations=6)
                row = {'arm': arm, 'draw_seed': draw_seed, 'draw_row': i,
                    'physical_section_id': section_id, 'animal_id': record['base_lineage']['animal_id'],
                    'specimen_id': record['base_lineage']['specimen_id'],
                    'experiment_id': record['base_lineage']['experiment_id'],
                    'virtual_subject_id': record['virtual_subject_id'], 'mode': record['mode'],
                    'raw_exterior_code': record['one_shot_slide_artifacts_v3']['parameters']['raw_exterior_code'],
                    'valid_fraction': float(valid192[i].mean()),
                    'proxy_tissue_fraction': float(tissue.mean()),
                    'interior_fraction': float(interior.mean()), 'interior_pixels': int(interior.sum()),
                    'proxy_precision_against_valid': float((tissue & valid192[i]).sum() / max(1, tissue.sum())),
                    'proxy_recall_against_valid': float((tissue & valid192[i]).sum() / max(1, valid192[i].sum())),
                    'provenance': record}
                if interior.sum() >= 1024:
                    contrast = float(np.quantile(pixels[interior], .9) - np.quantile(pixels[interior], .1))
                    high1 = pixels - ndi.gaussian_filter(pixels, 1)
                    high3 = pixels - ndi.gaussian_filter(pixels, 3)
                    row.update(interior_contrast_q90_q10=contrast,
                        interior_highpass_1px_rms=float(np.sqrt(np.mean(high1[interior] ** 2))),
                        interior_highpass_3px_rms=float(np.sqrt(np.mean(high3[interior] ** 2))),
                        interior_highpass_1px_over_contrast=float(np.sqrt(np.mean(high1[interior] ** 2)) / max(.01, contrast)),
                        interior_highpass_3px_over_contrast=float(np.sqrt(np.mean(high3[interior] ** 2)) / max(.01, contrast)))
                rows.append(row)

                mode = record['mode']
                if mode in target_angles:
                    normal = np.abs(record['virtual_unit_normal'])
                    angle = float(np.degrees(np.arccos(normal.max())))
                    rank = (abs(angle - target_angles[mode]), hashlib.sha256(section_id.encode()).hexdigest())
                    key = (arm, mode)
                    if key not in selected or rank < selected[key]['rank']:
                        selected[key] = {'rank': rank, 'row': row,
                            'state': batch['state'][i:i+1].cpu().clone(),
                            'reflection': int(batch['reflection'][i]),
                            'offsets': batch['offsets'][i:i+1].cpu().clone(),
                            'weights': batch['weights'][i:i+1].cpu().clone(),
                            'input': pixels.copy(), 'angle': angle}
            draw_seed += 1
            if accepted % 64 == 0:
                print(f'{arm} accepted {accepted}/256', flush=True)

assert len(rows) == 512 and len(selected) == 6
metrics = ('proxy_tissue_fraction', 'interior_fraction', 'interior_contrast_q90_q10',
    'interior_highpass_1px_rms', 'interior_highpass_3px_rms',
    'interior_highpass_1px_over_contrast', 'interior_highpass_3px_over_contrast')
groups = {f'{arm}_{name}': [row for row in rows if row['arm'] == arm and (
    name == 'all' or (name == 'raw_exact_black' and row['mode'] == 'raw' and row['raw_exterior_code'] == 0)
    or (name == 'other' and not (row['mode'] == 'raw' and row['raw_exterior_code'] == 0)))]
    for arm in ('v4', 'v5') for name in ('all', 'raw_exact_black', 'other')}
summary = {'scope': 'TRAIN synthetic only; separate unpaired random v4/v5 streams, 256 eligible physical sections each',
    'reference': 'frozen 119: 512 acquired TRAIN sections from 512 distinct donors; same 192-pixel proxy',
    'frozen_119_real': prior_summary['groups']['real'],
    'groups': {name: {'sections': len(group),
        'measurable': sum('interior_contrast_q90_q10' in row for row in group),
        'metrics': {metric: {'mean': float(np.mean([row[metric] for row in group if metric in row])),
            'q10_q50_q90': np.quantile([row[metric] for row in group if metric in row], [.1, .5, .9]).tolist()}
            for metric in metrics if any(metric in row for row in group)}} for name, group in groups.items()},
    'attempts': {arm: sum(row['arm'] == arm for row in attempts) for arm in ('v4', 'v5')},
    'exact_black_checked_pixels': {arm: sum(row['raw_exact_black_exterior_pixels_checked']
        for row in attempts if row['arm'] == arm) for arm in ('v4', 'v5')},
    'limitation': 'A threshold tissue proxy and scalar texture statistics cannot establish realistic anatomy, staining, or section defects.'}

figure, axes = plt.subplots(6, 2, figsize=(6, 15), facecolor='white')
grid_rows = []
axis = torch.arange(192, device='cuda') / 192
y, x = torch.meshgrid(axis, axis, indexing='ij')
with torch.inference_mode():
    for index, (arm, mode) in enumerate((arm, mode) for arm in ('v4', 'v5')
                                         for mode in ('raw', 'exact_black', 'imperfect_brush')):
        chosen = selected[(arm, mode)]
        centre, frame, basis = full_frame_state_to_components(chosen['state'].cuda())
        chart = torch.stack(((191 / 192 - x) if chosen['reflection'] else x, y), -1)
        plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        slab = plane[:, None] + chosen['offsets'].cuda()[:, :, None, None, None] * frame[:, None, None, None, :, 2]
        atlas = render_finite_thickness_coordinate_grid(context['atlas'], slab,
            (0., 0., 0.), (25., 25., 25.), chosen['weights'].cuda())[0]
        target = (atlas[0] / atlas[1].clamp_min(1e-4)).cpu().numpy()
        target[atlas[1].cpu().numpy() < .05] = 0
        input_image = chosen['input']
        axes[index, 0].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[index, 1].imshow(input_image, cmap='gray', vmin=0,
                              vmax=max(.001, float(np.quantile(input_image, .995))))
        axes[index, 0].set_ylabel(f'{arm}: {mode}\n{chosen["angle"]:.0f}° to cardinal', fontsize=9)
        for panel in axes[index]:
            panel.set_xticks([])
            panel.set_yticks([])
        grid_rows.append({'arm': arm, 'mode': mode, 'physical_section_id': chosen['row']['physical_section_id'],
            'draw_seed': chosen['row']['draw_seed'], 'draw_row': chosen['row']['draw_row'],
            'angle_to_cardinal_deg': chosen['angle'], 'selection': 'closest to predeclared mode angle; no image metric'})
axes[0, 0].set_title('atlas target gauge')
axes[0, 1].set_title('actual generated input; display-stretched')
figure.tight_layout()
out.mkdir(parents=True, exist_ok=False)
figure.savefig(out / 'atlas_input_grid.png', dpi=150)
plt.close(figure)
for name, data in (('rows.jsonl', rows), ('attempts.jsonl', attempts)):
    (out / name).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in data), encoding='utf-8')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
(out / 'grid_selection.json').write_text(json.dumps(grid_rows, indent=2) + '\n', encoding='utf-8')
sources = ('qc_v5_tissue_grain_135.py', 'arbitrary_plane_one_shot_slide_artifacts_v5.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py', 'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py')
receipt = {'rows': 512, 'distinct_attempted_physical_sections': len(seen),
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in sources},
    'protocol_sha256': hashlib.sha256((Path(__file__).parent.parent / 'docs/publication/V5_TISSUE_GRAIN_QC_135_PROTOCOL_20261010.md').read_bytes()).hexdigest(),
    'frozen_119_summary_sha256': prior_done['summary_sha256'],
    'frozen_119_completed_sha256': hashlib.sha256((prior / 'completed.json').read_bytes()).hexdigest(),
    'context_provenance_sha256': hashlib.sha256(json.dumps(context['provenance'], sort_keys=True).encode()).hexdigest(),
    'output_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ('rows.jsonl', 'attempts.jsonl', 'summary.json', 'grid_selection.json', 'atlas_input_grid.png')}}
(out / 'completed.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(out), 'sections_per_arm': 256,
    'raw_exact_black_highpass_1px': {arm: summary['groups'][f'{arm}_raw_exact_black']['metrics']
        ['interior_highpass_1px_over_contrast']['mean'] for arm in ('v4', 'v5')},
    'real_119_highpass_1px': prior_summary['groups']['real']['metrics']
        ['interior_highpass_1px_over_contrast']['mean']}, indent=2), flush=True)
