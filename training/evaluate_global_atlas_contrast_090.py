"""Frozen 086-beam DEV readout of 090 global image-atlas compatibility."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.global_atlas_contrast_090 import GlobalAtlasContrast090, render_atlas_planes_090

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GLOBAL_ATLAS_CONTRAST_090_PROTOCOL_20261004.md'
run = root / 'runs/global_atlas_contrast_090_pilot'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
rank_run = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
parent_run = root / 'runs/one_shot_joint_atlas_feedback_089_pilot'
previous = root / 'runs/one_shot_joint_atlas_feedback_089_development_eval'
out = root / 'runs/global_atlas_contrast_090_development_eval'
steps = (0, 3000, 8000, 15000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


training = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
assert training['protocol_sha256'] == sha(protocol)
assert finished['config_sha256'] == sha(run / 'config.json')
assert finished['draws_sha256'] == sha(run / 'draws.jsonl')
assert finished['training_sha256'] == sha(run / 'training.jsonl')
assert set(finished['checkpoint_sha256']) == set(map(str, steps))
assert all(finished['checkpoint_sha256'][str(step)] == sha(
    run / f'contrast_step_{step:05d}.pt') for step in steps)
assert finished['accepted_synthetic'] == training['batches'] == 15000
assert tuple(training['checkpoints']) == steps
assert training['parent_089_checkpoint_sha256'] == sha(
    parent_run / 'joint_step_06000.pt') == \
    'dd49c68416f406d9d9a75f54b235f9c0984cadf0aea0b462febb87e6fbd11f0a'
assert all(sha(source / name) == digest for name, digest in training['source_sha256'].items())
for receipt in (training, finished):
    assert not any(receipt[key] for key in ('calibrated', 'public_benchmark_used',
                                             'real_labels_used', 'external_pretrained_weights_used'))

accepted, attempts = 0, 0
for draw in map(json.loads, (run / 'draws.jsonl').open()):
    assert draw['step'] == accepted + 1
    assert draw['seed_prefix'][0] == training['seed'] * 1000000 + attempts
    assert draw['split'] == draw['base_lineage']['split'] == 'train'
    attempts += 1
    accepted += bool(draw['used'])
assert accepted == 15000 and attempts == finished['draw_attempts']

assert sha(panel / 'completed.json') == \
    '3fee767a1b33b83a13560a6766e38fa9502d967e54bbddd5aafd74994d5c5a8b'
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 246 and len({row['section_id'] for row in records}) == 246
assert all(sha(panel / row['file']) == row['sha256'] for row in records)

assert sha(rank_run / 'completed.json') == \
    '5eb6dc84b615f3ce01ee9ac84ce124057a42a6dda39dbc2b999c87c1e05b3711'
rank_receipt = json.loads((rank_run / 'completed.json').read_text())
assert sha(rank_run / 'candidates.jsonl') == rank_receipt['candidates_sha256']
assert sha(rank_run / 'sections.jsonl') == rank_receipt['sections_sha256']
assert sha(previous / 'completed.json') == \
    '7a53da6a26c53a38645dea3fb2ae13bc9da67a3f6443e01b8e46e3dce3aaffe1'
prior_receipt = json.loads((previous / 'completed.json').read_text())
assert sha(previous / 'candidates.jsonl') == prior_receipt['candidates_sha256'] == \
    'e3d1d032e519a1bd8c29fb4bf56c5a37fc3ec316da8cae716874f8b992e802c8'
assert sha(previous / 'sections.jsonl') == prior_receipt['sections_sha256'] == \
    '44db305d4ec17f0942007c161a733f6ce557a1ddd54af11219597d3eafce21c1'
assert sha(previous / 'summary.json') == prior_receipt['summary_sha256'] == \
    'cbcb1c4cea261c81814b79cd3143a981335cd2eadd90a60c4484721f28b85fdd'
previous_summary = json.loads((previous / 'summary.json').read_text())['6000']
previous_sections = {row['section_id']: row for row in
    map(json.loads, (previous / 'sections.jsonl').open()) if row['step'] == 6000}
frozen = {row['section_id']: [None] * 14 for row in records}
for row in map(json.loads, (previous / 'candidates.jsonl').open()):
    if row['step'] == 6000:
        slot = row['beam_slot']
        assert row['section_id'] in frozen and frozen[row['section_id']][slot] is None
        frozen[row['section_id']][slot] = row
ranked = {row['section_id']: [None] * 14 for row in records}
for row in map(json.loads, (rank_run / 'candidates.jsonl').open()):
    ranked[row['section_id']][row['beam_slot']] = row['branch_id']
assert len(previous_sections) == 246
for record in records:
    rows = frozen[record['section_id']]
    assert all(row is not None for row in rows)
    assert [row['branch_id'] for row in rows] == ranked[record['section_id']]
    assert all(row['source_sha256'] == record['sha256'] for row in rows)
    assert all(np.isfinite(row['mapped96_error_um']) for row in rows)

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
models = []
for step in steps:
    checkpoint = torch.load(run / f'contrast_step_{step:05d}.pt',
                            map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and json.loads(json.dumps(checkpoint['config'])) == training
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
    'checkpoint_sha256': finished['checkpoint_sha256'],
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'ranking_086_completed_sha256': sha(rank_run / 'completed.json'),
    'parent_089_completed_sha256': sha(parent_run / 'completed.json'),
    'previous_eval_completed_sha256': sha(previous / 'completed.json'),
    'previous_eval_candidates_sha256': sha(previous / 'candidates.jsonl'),
    'steps': steps, 'sections_per_step': 246, 'candidates_per_section': 14,
    'beam': 'exact 086 branch IDs with frozen 089 batch-6000 corrected states',
    'error': 'frozen 089 mapped96 surviving-pixel 3-D CCF error in micrometres',
    'aggregation': 'equal weight to each represented synthetic subject plan',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

sections = []
with torch.inference_mode(), (out / 'candidates.jsonl').open('w') as candidates_file, \
        (out / 'sections.jsonl').open('w') as sections_file:
    for index, record in enumerate(records, 1):
        rows = frozen[record['section_id']]
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        states = torch.tensor([[row['corrected_state'] for row in rows]], device='cuda')
        reflections = torch.tensor([[row['reflection'] for row in rows]], device='cuda')
        atlas_pair = render_atlas_planes_090(atlas, states, reflections, offsets, weights,
                                              candidate_chunk=2, source_shape=(256, 256))
        atlas_support = atlas_pair[0, :, 1].mean((-2, -1)).cpu().numpy()
        physical = np.array([row['mapped96_error_um'] for row in rows])
        branch_ids = [row['branch_id'] for row in rows]
        best_slot = int(physical.argmin())
        old_section = previous_sections[record['section_id']]
        for step, scorer in zip(steps, models):
            scores = scorer(image, atlas_pair, candidate_chunk=3)[0].cpu().numpy()
            assert scores.shape == (14,) and np.isfinite(scores).all()
            order = np.argsort(-scores, kind='stable')
            selected_slot = int(order[0])
            rho = (0. if np.ptp(scores) == 0 else
                   float(spearmanr(scores, -physical).statistic))
            shared = {key: record[key] for key in ('section_id', 'animal_id',
                'synthetic_animal_id', 'specimen_id', 'experiment_id',
                'synthetic_subject_plan_id', 'synthetic_subject_realization_id',
                'appearance_mode')}
            shared.update({'step': step, 'source_file': record['file'],
                           'source_sha256': record['sha256'],
                           'valid_pixels': record['valid_pixels'],
                           'visible_fraction': record['valid_pixels'] / 256**2,
                           'low_support': rows[0]['low_support']})
            for slot, row in enumerate(rows):
                candidates_file.write(json.dumps({**shared,
                    'beam_slot': slot, 'branch_id': row['branch_id'],
                    'reflection': row['reflection'], 'score': float(scores[slot]),
                    'mapped96_error_um': float(physical[slot]),
                    'atlas_support_mean': float(atlas_support[slot]),
                    'in_updated_prior_beam': row['in_updated_prior_beam'],
                    'selected': slot == selected_slot, 'best14': slot == best_slot}) + '\n')
            section = {**shared,
                'selected_slot': selected_slot,
                'selected_branch': branch_ids[selected_slot],
                'best14_branch': branch_ids[best_slot],
                'selected_mapped96_um': float(physical[selected_slot]),
                'best14_mapped96_um': float(physical[best_slot]),
                'best14_score_rank': int(np.flatnonzero(order == best_slot)[0] + 1),
                'selected_regret_um': float(physical[selected_slot] - physical[best_slot]),
                'score_error_spearman': rho,
                'selected_in_updated_prior_beam': rows[selected_slot]['in_updated_prior_beam'],
                'updated_prior_beam_overlap_count': old_section['updated_prior_beam_overlap_count'],
                'old_selected_mapped96_um': old_section['fitted_selected_mapped96_um'],
                'selected_gain_vs_089_um': float(old_section['fitted_selected_mapped96_um']
                                                 - physical[selected_slot])}
            for k in (1, 3, 5, 14):
                section[f'top{k}_best_mapped96_um'] = float(physical[order[:k]].min())
            sections.append(section)
            sections_file.write(json.dumps(section, allow_nan=False) + '\n')
        if index % 32 == 0 or index == len(records):
            print(json.dumps({'sections': index, 'total': len(records)}), flush=True)

metrics = ('selected_mapped96_um', 'best14_mapped96_um', 'selected_regret_um',
           'best14_score_rank', 'score_error_spearman', 'selected_in_updated_prior_beam',
           'updated_prior_beam_overlap_count', 'old_selected_mapped96_um',
           'selected_gain_vs_089_um', *(f'top{k}_best_mapped96_um' for k in (1, 3, 5, 14)))
summary = {}
for step in steps:
    group = [row for row in sections if row['step'] == step]
    groups = {'all': group,
              'low_support': [row for row in group if row['low_support']]}
    groups.update({name: [row for row in group if row['appearance_mode'] == name]
                   for name in ('exact_black', 'raw', 'imperfect_brush')})
    summary[str(step)] = {}
    for name, subset in groups.items():
        plans = sorted({row['synthetic_subject_plan_id'] for row in subset})
        by_plan = {plan: {'sections': sum(row['synthetic_subject_plan_id'] == plan
                                          for row in subset),
            **{metric: float(np.mean([row[metric] for row in subset
                if row['synthetic_subject_plan_id'] == plan])) for metric in metrics}}
            for plan in plans}
        summary[str(step)][name] = {'sections': len(subset), 'plans': len(plans),
            'by_plan': by_plan,
            'plan_equal_mean': {metric: float(np.mean([by_plan[plan][metric]
                for plan in plans])) for metric in metrics}}
    assert summary[str(step)]['all']['plans'] == 8

baseline = previous_summary['groups']['all']['plan_equal_mean']['fitted_selected_mapped96_um']
black_baseline = previous_summary['groups']['appearance_exact_black']['plan_equal_mean'][
    'fitted_selected_mapped96_um']
assert abs(baseline - 2591.92483804323) < 1e-6
assert abs(black_baseline - 2562.37015868252) < 1e-6
final = summary['15000']
summary['prespecified_gate'] = {
    'selected_gain_ge_400um': final['all']['plan_equal_mean']['selected_mapped96_um']
                               <= baseline - 400,
    'score_error_rho_ge_0_30': final['all']['plan_equal_mean']['score_error_spearman'] >= .30,
    'black_regression_le_150um': final['exact_black']['plan_equal_mean'][
        'selected_mapped96_um'] <= black_baseline + 150}
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
