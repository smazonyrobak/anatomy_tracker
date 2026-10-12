"""Model-free atlas-intensity comparison on frozen blind 270 proposal pairs."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090


PARENT = ROOT / 'runs/pose_capture_270_v9_continuation_001'
OUT = ROOT / 'runs/blind_atlas_pair_information_271_train_inner_001'
SOURCE = Path(__file__).resolve().parent
DOCUMENT = SOURCE.parent / 'docs/publication/BLIND_ATLAS_PAIR_INFORMATION_271_PROTOCOL_20261012.md'
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
done = json.loads((PARENT / 'completed.json').read_text())
assert done['candidate'] == 270 and done['added_optimizer_batches'] == 8000
selected_step = done['selected_step']
assert hashlib.sha256((PARENT / f'pose_step_{selected_step:05d}.pt').read_bytes()).hexdigest() == (
    done['checkpoint_sha256'][str(selected_step)])
panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64
draws = [json.loads(line) for line in (PARENT / 'inner_draws.jsonl').open()]
accepted = [draw for draw in draws if draw['used']]
assert len(accepted) == len(panel)
assert all(draw['provenance']['physical_section_id'] == item['physical_section_id']
           for draw, item in zip(accepted, panel, strict=True))
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
model.load_state_dict(torch.load(PARENT / f'pose_step_{selected_step:05d}.pt',
                                 map_location='cpu', weights_only=False)['model'])
source_names = ('assay_blind_atlas_pair_information_271.py',
                'global_atlas_contrast_090.py', 'arbitrary_plane_one_shot_model.py',
                'arbitrary_plane_one_shot_slide_artifacts_v9.py')
config = {'candidate': 271, 'selected_parent_step': selected_step,
    'parent_completed_sha256': hashlib.sha256((PARENT / 'completed.json').read_bytes()).hexdigest(),
    'parent_panel_sha256': done['inner_panel_sha256'],
    'parent_checkpoint_sha256': done['checkpoint_sha256'][str(selected_step)],
    'protocol_sha256': hashlib.sha256(DOCUMENT.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                      for name in source_names},
    'beam': 32, 'near_mm': 1.5, 'far_mm': 2.5,
    'score': 'absolute Pearson of 3x3-smoothed image and atlas on identical pairwise '
             '5x5-eroded common atlas support',
    'minimum_common_cells': 64, 'minimum_support_iou': .45,
    'maximum_support_area_fraction_gap': .15,
    'true_psf_used': True, 'truth_used_to_select_analysis_pairs_only': True,
    'calibrated': False, 'biological_validation': False, 'public_benchmark_used': False}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
pair_rows, case_rows = [], []
with torch.inference_mode(), (OUT / 'pairs.jsonl').open('w') as pairs_file, \
        (OUT / 'cases.jsonl').open('w') as cases_file:
    for index, (item, draw) in enumerate(zip(panel, accepted, strict=True)):
        replay = sample_one_shot_slide_artifacts_v9(
            context, [draw['subject_index']], draw['draw_seed'], side=192)
        assert replay['provenance'][0]['physical_section_id'] == item['physical_section_id']
        assert torch.allclose(replay['inputs'][0].cpu(), item['inputs'], atol=1e-6, rtol=0)
        image = item['inputs'][None].cuda()
        chart = item['chart'].cuda()
        true_state = item['state'][None].cuda()
        true_reflection = item['reflection'][None].cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
        order = prior.argsort(descending=True)[:32]
        states = prediction['state'][0, order // 2]
        reflections = order % 2
        truth = rigid_points_090(true_state, true_reflection, chart, (192, 192))
        points = rigid_points_090(states[None], reflections[None], chart[None], (192, 192))
        errors = ((points - truth[:, None]).norm(dim=-1).mean(-1) / 1000)[0]
        rendered = render_atlas_planes_090(
            context['atlas'], states[None], reflections[None], replay['offsets'],
            replay['weights'], candidate_chunk=2, source_shape=(192, 192))[0]
        observed = F.avg_pool2d(F.interpolate(image[:, :1], (96, 96), mode='area'),
                                3, 1, 1)[0, 0]
        atlas = F.avg_pool2d(rendered[:, :1], 3, 1, 1)[:, 0]
        null_images = []
        for branch in range(32):
            seed = int.from_bytes(hashlib.sha256(
                f"{item['physical_section_id']}|{int(order[branch])}".encode()).digest()[:8],
                'little') % (2**63)
            intensity = rendered[branch, 0].flatten().clone()
            inside = (rendered[branch, 1].flatten() > .5).nonzero().flatten()
            permutation = torch.randperm(inside.numel(), device='cuda',
                generator=torch.Generator(device='cuda').manual_seed(seed))
            intensity[inside] = intensity[inside][permutation]
            null_images.append(intensity.reshape(96, 96))
        null_atlas = F.avg_pool2d(torch.stack(null_images)[:, None], 3, 1, 1)[:, 0]
        case_pairs = []
        near = (errors <= 1.5).nonzero().flatten().tolist()
        far = (errors >= 2.5).nonzero().flatten().tolist()
        for good in near:
            for bad in far:
                support_good = rendered[good, 1] > .5
                support_bad = rendered[bad, 1] > .5
                shared = support_good & support_bad
                common = F.avg_pool2d(shared[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6
                cells = int(common.sum())
                union = int((support_good | support_bad).sum())
                iou = int(shared.sum()) / union if union else 0.
                area_gap = abs(int(support_good.sum()) - int(support_bad.sum())) / 96**2
                if cells < 64 or iou < .45 or area_gap > .15:
                    continue
                scores = []
                for candidate in (atlas[good], atlas[bad], null_atlas[good], null_atlas[bad]):
                    a, b = observed[common], candidate[common]
                    a, b = a - a.mean(), b - b.mean()
                    scores.append(float((a * b).mean().abs() /
                                        (a.square().mean() * b.square().mean() + 1e-16).sqrt()))
                row = {'physical_section_id': item['physical_section_id'],
                    'base_index': item['base_index'], 'slot': item['slot'],
                    'good_rank': good, 'bad_rank': bad,
                    'good_branch': int(order[good]), 'bad_branch': int(order[bad]),
                    'good_mm': float(errors[good]), 'bad_mm': float(errors[bad]),
                    'common_cells': cells, 'support_iou': iou,
                    'support_area_fraction_gap': area_gap,
                    'intact_good_score': scores[0], 'intact_bad_score': scores[1],
                    'null_good_score': scores[2], 'null_bad_score': scores[3],
                    'intact_win': scores[0] > scores[1], 'null_win': scores[2] > scores[3],
                    'prior_win': good < bad}
                case_pairs.append(row)
                pair_rows.append(row)
                pairs_file.write(json.dumps(row, allow_nan=False) + '\n')
        case = {'physical_section_id': item['physical_section_id'],
            'base_index': item['base_index'], 'slot': item['slot'],
            'visible_fraction': item['visible_fraction'], 'obliquity_deg': item['obliquity_deg'],
            'source_mode': item['source_mode'], 'exterior_style': item['exterior_style'],
            'near_top32': len(near), 'far_top32': len(far), 'eligible_pairs': len(case_pairs),
            'intact_concordance': float(np.mean([row['intact_win'] for row in case_pairs]))
            if case_pairs else None,
            'null_concordance': float(np.mean([row['null_win'] for row in case_pairs]))
            if case_pairs else None,
            'prior_concordance': float(np.mean([row['prior_win'] for row in case_pairs]))
            if case_pairs else None}
        case_rows.append(case)
        cases_file.write(json.dumps(case, allow_nan=False) + '\n')
        if (index + 1) % 8 == 0:
            pairs_file.flush()
            cases_file.flush()
            print(json.dumps({'event': 'case_milestone', 'cases': index + 1,
                'eligible_pairs': len(pair_rows)}), flush=True)

bases = {}
for base in range(56, 64):
    cases = [case for case in case_rows if case['base_index'] == base and case['eligible_pairs']]
    bases[str(base)] = {'eligible_cases': len(cases),
        'eligible_pairs': sum(case['eligible_pairs'] for case in cases),
        **{kind: float(np.mean([case[f'{kind}_concordance'] for case in cases]))
           if cases else None for kind in ('intact', 'null', 'prior')}}
eligible_bases = [base for base in bases.values() if base['eligible_cases']]
means = {kind: float(np.mean([base[kind] for base in eligible_bases]))
         if eligible_bases else None for kind in ('intact', 'null', 'prior')}
gate = (len(pair_rows) >= 200 and len(eligible_bases) >= 6 and means['intact'] >= .7 and
        means['intact'] - means['null'] >= .15 and
        means['intact'] - means['prior'] >= .10)
summary = {'cases': len(case_rows), 'near_top32_cases': sum(case['near_top32'] > 0 for case in case_rows),
    'eligible_pairs': len(pair_rows), 'eligible_bases': len(eligible_bases),
    'equal_base_concordance': means, 'bases': bases, 'information_gate': gate,
    'interpretation': 'truth-labelled analysis pairs, not blind selection or calibrated posterior'}
summary['strata'] = {}
for label, cases in (
    ('visible_lt_0.15', [case for case in case_rows if case['visible_fraction'] < .15]),
    ('visible_ge_0.15', [case for case in case_rows if case['visible_fraction'] >= .15]),
    ('axis_le_20deg', [case for case in case_rows if case['obliquity_deg'] <= 20]),
    ('axis_20_to_45deg', [case for case in case_rows if 20 < case['obliquity_deg'] <= 45]),
    ('axis_gt_45deg', [case for case in case_rows if case['obliquity_deg'] > 45])):
    eligible = [case for case in cases if case['eligible_pairs']]
    summary['strata'][label] = {'cases': len(cases), 'eligible_cases': len(eligible),
        'eligible_pairs': sum(case['eligible_pairs'] for case in eligible),
        **{kind: float(np.mean([case[f'{kind}_concordance'] for case in eligible]))
           if eligible else None for kind in ('intact', 'null', 'prior')}}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(OUT / 'completed.json').write_text(json.dumps({
    'cases': len(case_rows), 'pairs': len(pair_rows), 'information_gate': gate,
    'config_sha256': hashlib.sha256((OUT / 'config.json').read_bytes()).hexdigest(),
    'cases_sha256': hashlib.sha256((OUT / 'cases.jsonl').read_bytes()).hexdigest(),
    'pairs_sha256': hashlib.sha256((OUT / 'pairs.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'eligible_pairs': len(pair_rows),
    'equal_base_concordance': means, 'information_gate': gate}), flush=True)
