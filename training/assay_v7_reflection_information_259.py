"""Read-only mirror identifiability on the frozen v7 synthetic DEV panel."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
OUT = ROOT / 'runs/v7_reflection_information_259_dev_001'
DOC = Path('docs/publication/V7_REFLECTION_INFORMATION_259_PROTOCOL_20261012.md')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
eligible = [row for row in records if row['eligible'] and row['valid_pixels'] > 0]
assert len(eligible) == 227 and len({row['provenance']['physical_section_id'] for row in eligible}) == 227
assert len({row['animal_id'] for row in eligible}) == 8

context = load_streaming_synthetic_v7_64(device='cuda')
config = {'scope': 'read-only frozen 248 synthetic DEV reflection information at true plane',
          'score': '257/180 absolute Pearson after 3x3 smoothing on 5x5-eroded shared atlas support',
          'minimum_common_cells': 64, 'minimum_support_iou': .45,
          'maximum_support_area_fraction_gap': .15,
          'null': 'candidate-specific deterministic within-support atlas-intensity permutation',
          'observed_tissue_mask_used': False, 'true_pose_used': True,
          'learned_weights_used': False, 'calibrated': False, 'public_benchmark_used': False,
          'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
          'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
          'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                            for name in (Path(__file__).name, 'global_atlas_contrast_090.py')},
          'atlas_provenance': context['provenance']}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        identity = record['provenance']['physical_section_id']
        assert hashlib.sha256((PANEL / record['file']).read_bytes()).hexdigest() == record['sha256']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][0:1][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        states = state[:, None].expand(-1, 2, -1)
        flags = torch.tensor(((reflection, 1 - reflection),), device='cuda')
        rendered = render_atlas_planes_090(context['atlas'], states, flags,
                                           offsets, weights, source_shape=(192, 192))[0]
        support_a, support_b = rendered[:, 1] > .5
        shared = support_a & support_b
        common = F.avg_pool2d(shared[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6
        cells = int(common.sum())
        union = int((support_a | support_b).sum())
        iou = int(shared.sum()) / union if union else 0.
        area_gap = abs(int(support_a.sum()) - int(support_b.sum())) / 96**2
        used = cells >= 64 and iou >= .45 and area_gap <= .15
        observed = F.avg_pool2d(F.interpolate(image, (96, 96), mode='area'), 3, 1, 1)[0, 0]
        atlas = F.avg_pool2d(rendered[:, :1], 3, 1, 1)[:, 0]
        null = []
        for branch in range(2):
            seed = int.from_bytes(hashlib.sha256(f'{identity}|reflection|{branch}'.encode()).digest()[:8],
                                  'little') % (2**63)
            intensity = rendered[branch, 0].flatten().clone()
            inside = (rendered[branch, 1].flatten() > .5).nonzero().flatten()
            order = torch.randperm(inside.numel(), device='cuda',
                                   generator=torch.Generator(device='cuda').manual_seed(seed))
            intensity[inside] = intensity[inside][order]
            null.append(intensity.reshape(96, 96))
        null_atlas = F.avg_pool2d(torch.stack(null)[:, None], 3, 1, 1)[:, 0]
        scores = []
        if cells >= 2:
            for candidate in (atlas[0], atlas[1], null_atlas[0], null_atlas[1]):
                a, b = observed[common], candidate[common]
                a, b = a - a.mean(), b - b.mean()
                scores.append(float((a * b).mean().abs() /
                                    (a.square().mean() * b.square().mean() + 1e-16).sqrt()))
        else:
            scores = [None] * 4
        row = {'physical_section_id': identity, 'animal_id': record['animal_id'],
               'appearance_mode': record['appearance_mode'],
               'valid_fraction': record['valid_pixels'] / 192**2,
               'panel_file_sha256': record['sha256'], 'true_reflection': reflection,
               'common_cells': cells, 'support_iou': iou,
               'support_area_fraction_gap': area_gap, 'eligible': used,
               'true_score': scores[0], 'opposite_score': scores[1],
               'null_true_score': scores[2], 'null_opposite_score': scores[3],
               'intact_margin': scores[0] - scores[1] if cells >= 2 else None,
               'null_margin': scores[2] - scores[3] if cells >= 2 else None,
               'intact_win': bool(scores[0] > scores[1]) if used else None,
               'null_win': bool(scores[2] > scores[3]) if used else None}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 64 == 0:
            stream.flush()
            print(json.dumps({'sections': len(rows), 'eligible': len(eligible)}), flush=True)

used = [row for row in rows if row['eligible']]
groups = []
for animal in sorted({row['animal_id'] for row in rows}):
    group = [row for row in used if row['animal_id'] == animal]
    if group:
        groups.append((np.mean([row['intact_win'] for row in group]),
                       np.mean([row['null_win'] for row in group])))
intact = float(np.mean([row[0] for row in groups]))
null = float(np.mean([row[1] for row in groups]))
interpretation = ('substantial_information' if len(groups) == 8 and intact >= .80 and intact - null >= .20
                  else 'weak_under_this_score' if intact <= .65 else 'inconclusive')
summary = {'sections': len(rows), 'eligible_sections': len(used), 'eligible_groups': len(groups),
           'equal_group_true_reflection_win': intact, 'equal_group_null_win': null,
           'interpretation': interpretation, 'appearance': {}, 'visible': {}}
for mode in sorted({row['appearance_mode'] for row in rows}):
    group = [row for row in used if row['appearance_mode'] == mode]
    summary['appearance'][mode] = {'sections': len(group),
        'true_reflection_win': float(np.mean([row['intact_win'] for row in group])) if group else None,
        'null_win': float(np.mean([row['null_win'] for row in group])) if group else None}
for low, high in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01)):
    group = [row for row in used if low <= row['valid_fraction'] < high]
    summary['visible'][f'{low}_{high}'] = {'sections': len(group),
        'true_reflection_win': float(np.mean([row['intact_win'] for row in group])) if group else None}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'sections': len(rows), 'eligible_sections': len(used),
    'config_sha256': hashlib.sha256((OUT / 'config.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'interpretation': interpretation, 'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'eligible': len(used),
                  'interpretation': interpretation}), flush=True)
