"""Independent frozen-beam DEV readout of blind-pair rank continuation 091."""
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
os.environ['TRITON_CACHE_DIR'] = str(root / 'cache/triton')
sys.dont_write_bytecode = True

import numpy as np
import torch
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.global_atlas_contrast_090 import GlobalAtlasContrast090, render_atlas_planes_090

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GLOBAL_ATLAS_BLIND_RANK_091_PROTOCOL_20261004.md'
run = root / 'runs/global_atlas_blind_rank_091_pilot'
parent = root / 'runs/global_atlas_contrast_090_pilot'
parent_eval = root / 'runs/global_atlas_contrast_090_development_eval'
baseline = root / 'runs/one_shot_joint_atlas_feedback_089_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
out = root / 'runs/global_atlas_blind_rank_091_development_eval'
steps = (0, 5000, 15000, 30000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((run / 'config.json').read_text())
receipt = json.loads((run / 'completed.json').read_text())
assert config['protocol_sha256'] == sha(protocol)
assert receipt['config_sha256'] == sha(run / 'config.json')
assert receipt['draws_sha256'] == sha(run / 'draws.jsonl')
assert receipt['training_sha256'] == sha(run / 'training.jsonl')
assert tuple(config['checkpoints']) == steps
assert receipt['batches'] == receipt['accepted_synthetic'] == config['batches'] == 30000
assert set(receipt['checkpoint_sha256']) == set(map(str, steps))
assert all(sha(run / f'rank_step_{step:05d}.pt') ==
           receipt['checkpoint_sha256'][str(step)] for step in steps)
assert sha(parent / 'completed.json') == config['parent_090_completed_sha256'] == \
    'd547c347871f8f111e9405aca38d8b75dcc20ec1fb48219a6519ab3ee93262b2'
assert sha(parent / 'contrast_step_15000.pt') == config['parent_090_checkpoint_sha256'] == \
    'dc18a41478a628d5815a5da4ddca290c3700e47ec2547b84e50c063fc157e9dc'
assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
for item in (config, receipt):
    assert not any(item[key] for key in ('calibrated', 'public_benchmark_used',
                                         'real_labels_used', 'external_pretrained_weights_used'))
accepted, attempts = 0, 0
for draw in map(json.loads, (run / 'draws.jsonl').open()):
    assert draw['step'] == accepted + 1
    assert draw['split'] == draw['base_lineage']['split'] == 'train'
    assert draw['seed_prefix'][0] == config['seed'] * 1000000 + attempts
    attempts += 1
    accepted += bool(draw['used'])
assert accepted == 30000 and attempts == receipt['draw_attempts']

assert sha(panel / 'completed.json') == \
    '3fee767a1b33b83a13560a6766e38fa9502d967e54bbddd5aafd74994d5c5a8b'
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 246 and len({row['section_id'] for row in records}) == 246
assert all(sha(panel / row['file']) == row['sha256'] for row in records)

assert sha(baseline / 'completed.json') == \
    '7a53da6a26c53a38645dea3fb2ae13bc9da67a3f6443e01b8e46e3dce3aaffe1'
assert sha(parent_eval / 'completed.json') == \
    '8c56d54a299ebd351782401ca460302dd6ac1a5623e92e2293abdfdcc6f5c9ac'
for directory in (baseline, parent_eval):
    frozen_receipt = json.loads((directory / 'completed.json').read_text())
    for stem in ('config', 'candidates', 'sections', 'summary'):
        extension = 'jsonl' if stem in ('candidates', 'sections') else 'json'
        assert sha(directory / f'{stem}.{extension}') == frozen_receipt[f'{stem}_sha256']
old = {row['section_id']: row for row in map(json.loads, (baseline / 'sections.jsonl').open())
       if row['step'] == 6000}
frozen = {record['section_id']: [None] * 14 for record in records}
for row in map(json.loads, (baseline / 'candidates.jsonl').open()):
    if row['step'] == 6000:
        frozen[row['section_id']][row['beam_slot']] = row
prior_090 = {record['section_id']: [None] * 14 for record in records}
for row in map(json.loads, (parent_eval / 'candidates.jsonl').open()):
    if row['step'] == 15000:
        prior_090[row['section_id']][row['beam_slot']] = row
prior_090_sections = {row['section_id']: row for row in
    map(json.loads, (parent_eval / 'sections.jsonl').open()) if row['step'] == 15000}
assert len(old) == 246
assert len(prior_090_sections) == 246
for record in records:
    prior = frozen[record['section_id']]
    previous = prior_090[record['section_id']]
    assert all(row is not None for row in prior + previous)
    assert [row['branch_id'] for row in prior] == [row['branch_id'] for row in previous]
    assert all(row['source_sha256'] == record['sha256'] for row in prior + previous)
    assert all(abs(a['mapped96_error_um'] - b['mapped96_error_um']) < 1e-5
               for a, b in zip(prior, previous))

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
models = []
for step in steps:
    checkpoint = torch.load(run / f'rank_step_{step:05d}.pt',
                            map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step
    assert json.loads(json.dumps(checkpoint['config'])) == config
    scorer = GlobalAtlasContrast090().cuda().eval()
    scorer.load_state_dict(checkpoint['scorer'], strict=True)
    scorer.requires_grad_(False)
    models.append(scorer)
    del checkpoint

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
    'training_completed_sha256': sha(run / 'completed.json'),
    'training_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': receipt['checkpoint_sha256'],
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'baseline_089_completed_sha256': sha(baseline / 'completed.json'),
    'parent_090_eval_completed_sha256': sha(parent_eval / 'completed.json'),
    'beam': 'same 086 14 branch IDs, 089 batch-6000 corrected states and mapped96 errors',
    'aggregation': 'per-plan section mean, then equal mean of eight synthetic plans',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

sections = []
with torch.inference_mode(), (out / 'candidates.jsonl').open('w') as candidate_file, \
        (out / 'sections.jsonl').open('w') as section_file:
    for index, record in enumerate(records, 1):
        rows = frozen[record['section_id']]
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        states = torch.tensor([[row['corrected_state'] for row in rows]], device='cuda')
        reflection = torch.tensor([[row['reflection'] for row in rows]], device='cuda')
        pairs = render_atlas_planes_090(atlas, states, reflection, offsets, weights,
                                        candidate_chunk=2, source_shape=(256, 256))
        support = pairs[0, :, 1].mean((-2, -1)).cpu().numpy()
        physical = np.array([row['mapped96_error_um'] for row in rows])
        best = int(physical.argmin())
        previous = prior_090[record['section_id']]
        shared = {key: record[key] for key in ('section_id', 'animal_id',
            'synthetic_animal_id', 'specimen_id', 'experiment_id',
            'synthetic_subject_plan_id', 'synthetic_subject_realization_id',
            'appearance_mode')}
        shared.update({'source_file': record['file'], 'source_sha256': record['sha256'],
                       'valid_pixels': record['valid_pixels'],
                       'visible_fraction': record['valid_pixels'] / 256**2,
                       'low_support': rows[0]['low_support']})
        for step, scorer in zip(steps, models):
            scores = scorer(image, pairs, candidate_chunk=3)[0].cpu().numpy()
            assert scores.shape == (14,) and np.isfinite(scores).all()
            if step == 0:
                assert np.allclose(scores, [row['score'] for row in previous],
                                   rtol=1e-5, atol=1e-4)
                assert np.allclose(support, [row['atlas_support_mean'] for row in previous],
                                   rtol=1e-5, atol=1e-5)
            order = np.argsort(-scores, kind='stable')
            selected = int(order[0])
            rho = spearmanr(scores, -physical).statistic
            support_rho = spearmanr(scores, support).statistic
            assert np.isfinite(rho) and np.isfinite(support_rho)
            for slot, row in enumerate(rows):
                candidate_file.write(json.dumps({**shared, 'step': step,
                    'beam_slot': slot, 'branch_id': row['branch_id'],
                    'score': float(scores[slot]), 'atlas_support_mean': float(support[slot]),
                    'mapped96_error_um': float(physical[slot]),
                    'in_updated_prior_beam': row['in_updated_prior_beam'],
                    'selected': slot == selected, 'best14': slot == best}) + '\n')
            section = {**shared, 'step': step, 'selected_slot': selected,
                'selected_branch': rows[selected]['branch_id'],
                'best14_branch': rows[best]['branch_id'],
                'selected_mapped96_um': float(physical[selected]),
                'best14_mapped96_um': float(physical[best]),
                'selected_regret_um': float(physical[selected] - physical[best]),
                'best14_score_rank': int(np.flatnonzero(order == best)[0] + 1),
                'score_error_spearman': float(rho),
                'score_support_spearman': float(support_rho),
                'selected_support_mean': float(support[selected]),
                'best14_support_mean': float(support[best]),
                'selected_in_updated_prior_beam': rows[selected]['in_updated_prior_beam'],
                'updated_prior_beam_overlap_count': old[record['section_id']][
                    'updated_prior_beam_overlap_count'],
                'baseline_089_selected_mapped96_um': old[record['section_id']][
                    'fitted_selected_mapped96_um'],
                'baseline_090_selected_mapped96_um': prior_090_sections[record['section_id']][
                    'selected_mapped96_um']}
            for k in (1, 3, 5, 14):
                section[f'top{k}_best_mapped96_um'] = float(physical[order[:k]].min())
            sections.append(section)
            section_file.write(json.dumps(section, allow_nan=False) + '\n')
        if index % 32 == 0 or index == len(records):
            print(json.dumps({'sections': index, 'total': len(records)}), flush=True)

metrics = ('selected_mapped96_um', 'best14_mapped96_um', 'selected_regret_um',
           'best14_score_rank', 'score_error_spearman', 'score_support_spearman',
           'selected_support_mean', 'best14_support_mean',
           'selected_in_updated_prior_beam', 'updated_prior_beam_overlap_count',
           'baseline_089_selected_mapped96_um', 'baseline_090_selected_mapped96_um',
           *(f'top{k}_best_mapped96_um' for k in (1, 3, 5, 14)))
summary = {}
for step in steps:
    group = [row for row in sections if row['step'] == step]
    subsets = {'all': group, 'low_support': [row for row in group if row['low_support']]}
    subsets.update({name: [row for row in group if row['appearance_mode'] == name]
                    for name in ('exact_black', 'raw', 'imperfect_brush')})
    summary[str(step)] = {}
    for name, subset in subsets.items():
        plans = sorted({row['synthetic_subject_plan_id'] for row in subset})
        by_plan = {plan: {'sections': sum(row['synthetic_subject_plan_id'] == plan
                                          for row in subset),
            **{metric: float(np.mean([row[metric] for row in subset
                if row['synthetic_subject_plan_id'] == plan])) for metric in metrics}}
            for plan in plans}
        summary[str(step)][name] = {'sections': len(subset), 'plans': len(plans),
            'by_plan': by_plan,
            'plan_equal_mean': {metric: float(np.mean([by_plan[p][metric] for p in plans]))
                                for metric in metrics}}
    assert summary[str(step)]['all']['plans'] == 8

final = summary['30000']
summary['prespecified_gate'] = {
    'selected_gain_ge_400um': final['all']['plan_equal_mean']['selected_mapped96_um']
                               <= 2591.92483804323 - 400,
    'score_error_rho_ge_0_30': final['all']['plan_equal_mean']['score_error_spearman'] >= .30,
    'black_regression_le_150um': final['exact_black']['plan_equal_mean'][
        'selected_mapped96_um'] <= 2562.37015868252 + 150}
summary['prespecified_gate']['pass'] = all(summary['prespecified_gate'].values())
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
assert len(sections) == 246 * len(steps)
(out / 'completed.json').write_text(json.dumps({
    'sections': len(sections), 'candidates': len(sections) * 14,
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
