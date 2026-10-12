"""Frozen, model-free local atlas-information assay on the 248 v7 DEV panel."""
import hashlib
import json
import math
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

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
OUT = ROOT / 'runs/v7_atlas_information_257_dev_001'
DOC = Path('docs/publication/V7_ATLAS_INFORMATION_257_PROTOCOL_20261012.md')
NAMES = ('normal_plus_1p5', 'normal_minus_1p5', 'normal_plus_3', 'normal_minus_3',
         'u_tilt_plus_15', 'u_tilt_minus_15', 'v_tilt_plus_15', 'v_tilt_minus_15',
         'u_centre_plus_1p5', 'u_centre_minus_1p5', 'v_centre_plus_1p5', 'v_centre_minus_1p5',
         'roll_plus_15', 'roll_minus_15', 'u_span_times_1p2', 'u_span_div_1p2',
         'v_span_times_1p2', 'v_span_div_1p2', 'shear_plus_0p2', 'shear_minus_0p2')
KINDS = ('offset',) * 4 + ('tilt',) * 4 + ('centre',) * 4 + ('roll',) * 2 + ('span',) * 4 + ('shear',) * 2
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
eligible = [record for record in records if record['eligible'] and record['valid_pixels'] > 0]
assert len(records) == 256 and len(eligible) == 227
assert len({record['provenance']['physical_section_id'] for record in eligible}) == 227
assert len({record['animal_id'] for record in eligible}) == 8

context = load_streaming_synthetic_v7_64(device='cuda')
updates = torch.zeros((len(NAMES), 9), device='cuda')
updates[:4, 5] = torch.tensor((1500., -1500., 3000., -3000.), device='cuda')
updates[4:6, 0] = torch.tensor((math.pi / 12, -math.pi / 12), device='cuda')
updates[6:8, 1] = torch.tensor((math.pi / 12, -math.pi / 12), device='cuda')
updates[8:10, 3] = torch.tensor((1500., -1500.), device='cuda')
updates[10:12, 4] = torch.tensor((1500., -1500.), device='cuda')
updates[12:14, 2] = torch.tensor((math.pi / 12, -math.pi / 12), device='cuda')
updates[14:16, 6] = torch.tensor((math.log(1.2), -math.log(1.2)), device='cuda')
updates[16:18, 7] = torch.tensor((math.log(1.2), -math.log(1.2)), device='cuda')
updates[18:20, 8] = torch.tensor((.2, -.2), device='cuda')
config = {
    'scope': 'frozen 248 synthetic DEV local image-information assay, no model fitting',
    'cases': 227, 'candidate_bank': NAMES, 'candidate_types': KINDS,
    'source_side': 192, 'score_side': 96,
    'score': 'absolute Pearson after 3x3 smoothing on 5x5-eroded common atlas support',
    'minimum_common_cells': 64, 'minimum_support_iou': .45,
    'maximum_support_area_fraction_gap': .15,
    'null': 'deterministic independent within-atlas-support permutation per candidate',
    'observed_tissue_mask_used': False, 'true_pose_and_reflection_used': True,
    'learned_weights_used': False, 'calibrated': False, 'public_benchmark_used': False,
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in (Path(__file__).name, 'global_atlas_contrast_090.py',
                                   'arbitrary_plane_full_frame_primitives.py')},
    'atlas_provenance': context['provenance'],
}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

pair_rows, section_rows = [], []
with torch.inference_mode(), (OUT / 'pairs.jsonl').open('w', encoding='utf8') as pair_file, \
     (OUT / 'sections.jsonl').open('w', encoding='utf8') as section_file:
    for record in eligible:
        section_id = record['provenance']['physical_section_id']
        assert hashlib.sha256((PANEL / record['file']).read_bytes()).hexdigest() == record['sha256']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][0:1][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        states = torch.cat((state, compose_full_frame_state(state.expand(len(NAMES), -1), updates)), 0)
        flags = torch.full((1, len(states)), reflection, dtype=torch.long, device='cuda')
        rendered = render_atlas_planes_090(context['atlas'], states[None], flags,
                                           offsets, weights, source_shape=(192, 192))[0]
        observed = F.avg_pool2d(F.interpolate(image, (96, 96), mode='area'), 3, 1, 1)[0, 0]
        atlas = F.avg_pool2d(rendered[:, :1], 3, 1, 1)[:, 0]
        null_images = []
        for branch in range(len(states)):
            seed = int.from_bytes(hashlib.sha256(f'{section_id}|{branch}'.encode()).digest()[:8],
                                  'little') % (2**63)
            intensity = rendered[branch, 0].flatten().clone()
            inside = (rendered[branch, 1].flatten() > .5).nonzero().flatten()
            order = torch.randperm(inside.numel(), device='cuda',
                                   generator=torch.Generator(device='cuda').manual_seed(seed))
            intensity[inside] = intensity[inside][order]
            null_images.append(intensity.reshape(96, 96))
        null_atlas = F.avg_pool2d(torch.stack(null_images)[:, None], 3, 1, 1)[:, 0]
        section = {'physical_section_id': section_id, 'animal_id': record['animal_id'],
                   'appearance_mode': record['appearance_mode'],
                   'valid_fraction': record['valid_pixels'] / 192**2,
                   'panel_file_sha256': record['sha256'], 'eligible_pairs': 0,
                   'intact_wins': 0, 'null_wins': 0,
                   **{kind: {'eligible': 0, 'intact_wins': 0, 'null_wins': 0}
                      for kind in sorted(set(KINDS))}}
        for branch, name in enumerate(NAMES, 1):
            positive_support = rendered[0, 1] > .5
            wrong_support = rendered[branch, 1] > .5
            shared = positive_support & wrong_support
            common = F.avg_pool2d(shared[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6
            cells = int(common.sum())
            union = int((positive_support | wrong_support).sum())
            iou = int(shared.sum()) / union if union else 0.
            area_gap = abs(int(positive_support.sum()) - int(wrong_support.sum())) / 96**2
            used = cells >= 64 and iou >= .45 and area_gap <= .15
            scores = []
            if cells >= 2:
                for candidate in (atlas[0], atlas[branch], null_atlas[0], null_atlas[branch]):
                    a, b = observed[common], candidate[common]
                    a, b = a - a.mean(), b - b.mean()
                    scores.append(float((a * b).mean().abs() /
                                        (a.square().mean() * b.square().mean() + 1e-16).sqrt()))
            else:
                scores = [None] * 4
            kind = KINDS[branch - 1]
            row = {'physical_section_id': section_id, 'animal_id': record['animal_id'],
                   'appearance_mode': record['appearance_mode'], 'candidate': name,
                   'kind': kind, 'common_cells': cells, 'support_iou': iou,
                   'support_area_fraction_gap': area_gap, 'eligible': used,
                   'intact_positive': scores[0], 'intact_wrong': scores[1],
                   'null_positive': scores[2], 'null_wrong': scores[3],
                   'intact_margin': scores[0] - scores[1] if cells >= 2 else None,
                   'null_margin': scores[2] - scores[3] if cells >= 2 else None,
                   'intact_win': bool(scores[0] > scores[1]) if used else None,
                   'null_win': bool(scores[2] > scores[3]) if used else None}
            pair_rows.append(row)
            pair_file.write(json.dumps(row) + '\n')
            if used:
                section['eligible_pairs'] += 1
                section['intact_wins'] += row['intact_win']
                section['null_wins'] += row['null_win']
                section[kind]['eligible'] += 1
                section[kind]['intact_wins'] += row['intact_win']
                section[kind]['null_wins'] += row['null_win']
        section_rows.append(section)
        section_file.write(json.dumps(section) + '\n')
        if len(section_rows) % 64 == 0:
            pair_file.flush()
            section_file.flush()
            print(json.dumps({'sections': len(section_rows), 'eligible': len(eligible)}), flush=True)

summary = {'sections': len(section_rows), 'pairs': len(pair_rows), 'cohorts': {},
           'appearance': {}, 'gate': None,
           'interpretation': 'privileged true-pose local information only; no inference or biological validation'}
for kind in ('all', *sorted(set(KINDS))):
    groups = []
    for animal in sorted({row['animal_id'] for row in section_rows}):
        values = []
        for row in section_rows:
            if row['animal_id'] != animal:
                continue
            counts = row if kind == 'all' else row[kind]
            denominator = counts['eligible_pairs'] if kind == 'all' else counts['eligible']
            if denominator:
                values.append((counts['intact_wins'] / denominator,
                               counts['null_wins'] / denominator))
        if values:
            groups.append(np.mean(values, axis=0))
    pairs = [row for row in pair_rows if kind == 'all' or row['kind'] == kind]
    summary['cohorts'][kind] = {
        'eligible_pairs': sum(row['eligible'] for row in pairs),
        'eligible_groups': len(groups),
        'equal_group_intact_win': float(np.mean([row[0] for row in groups])) if groups else None,
        'equal_group_null_win': float(np.mean([row[1] for row in groups])) if groups else None,
    }
for mode in sorted({row['appearance_mode'] for row in section_rows}):
    cases = [row for row in section_rows if row['appearance_mode'] == mode and row['eligible_pairs']]
    summary['appearance'][mode] = {'sections': len(cases),
        'mean_intact_win': float(np.mean([row['intact_wins'] / row['eligible_pairs'] for row in cases])) if cases else None,
        'mean_null_win': float(np.mean([row['null_wins'] / row['eligible_pairs'] for row in cases])) if cases else None}
summary['visible'] = {}
for low, high in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01)):
    cases = [row for row in section_rows if low <= row['valid_fraction'] < high and row['eligible_pairs']]
    summary['visible'][f'{low}_{high}'] = {'sections': len(cases),
        'mean_intact_win': float(np.mean([row['intact_wins'] / row['eligible_pairs'] for row in cases])) if cases else None,
        'mean_null_win': float(np.mean([row['null_wins'] / row['eligible_pairs'] for row in cases])) if cases else None}
summary['gate'] = (summary['cohorts']['all']['eligible_pairs'] >= 500 and
                   all(summary['cohorts'][kind]['eligible_groups'] == 8 and
                       summary['cohorts'][kind]['eligible_pairs'] >= 50 and
                       summary['cohorts'][kind]['equal_group_intact_win'] >= .75 and
                       summary['cohorts'][kind]['equal_group_intact_win'] -
                       summary['cohorts'][kind]['equal_group_null_win'] >= .15
                       for kind in sorted(set(KINDS))))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'sections': len(section_rows), 'pairs': len(pair_rows),
    'config_sha256': hashlib.sha256((OUT / 'config.json').read_bytes()).hexdigest(),
    'sections_sha256': hashlib.sha256((OUT / 'sections.jsonl').read_bytes()).hexdigest(),
    'pairs_sha256': hashlib.sha256((OUT / 'pairs.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'gate': summary['gate'], 'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'sections': len(section_rows),
                  'eligible_pairs': summary['cohorts']['all']['eligible_pairs'],
                  'gate': summary['gate']}), flush=True)
