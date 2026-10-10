"""Frozen original-branch DEV selection for the matched evidence arms."""

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import nn

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_plane_evidence_130 import global_plane_evidence_130
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert root.drive.upper() == Path(__file__).resolve().drive.upper() == 'I:'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = load_streaming_synthetic_v7_64(device='cuda')['atlas']
panel = root / 'data/joint_in_path_correspondence_128_dev_panel'
panel_done = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
records = {row['section_id']: row for row in records if row['eligible']}
prior_eval = root / 'runs/joint_in_path_correspondence_128_dev_eval'
prior_done = json.loads((prior_eval / 'completed.json').read_text())
assert sha(prior_eval / 'synthetic_rows.jsonl') == prior_done['output_sha256']['synthetic_rows.jsonl']
prior_rows = [json.loads(line) for line in (prior_eval / 'synthetic_rows.jsonl').open()]
assert len(prior_rows) == len(records) == 243
assert set(records) == {row['section_id'] for row in prior_rows}
for record in records.values():
    assert sha(panel / record['file']) == record['sha256']

run_receipts = {}
configs = {}
draws = {}
for arm in ('full', 'support'):
    run = root / f'runs/crossed_plane_evidence_130_{arm}'
    run_receipts[arm] = json.loads((run / 'completed.json').read_text())
    assert sha(run / 'config.json') == run_receipts[arm]['config_sha256']
    assert sha(run / 'training.jsonl') == run_receipts[arm]['training_sha256']
    configs[arm] = json.loads((run / 'config.json').read_text())
    draws[arm] = [json.loads(line)['draws'] for line in
                  (run / 'training.jsonl').open()]
    assert len(draws[arm]) == configs[arm]['updates'] == 2000
assert {k: v for k, v in configs['full'].items() if k != 'arm'} == {
    k: v for k, v in configs['support'].items() if k != 'arm'}
assert draws['full'] == draws['support']

output = root / 'runs/crossed_plane_evidence_130_dev_eval'
assert not output.exists()
output.mkdir(parents=True)
all_rows = []
summary = {}
with torch.inference_mode():
    for arm in ('full', 'support'):
        run = root / f'runs/crossed_plane_evidence_130_{arm}'
        done = run_receipts[arm]
        assert done['updates'] == 2000
        for step in (500, 1000, 2000):
            checkpoint = run / f'joint_step_{step:05d}.pt'
            assert sha(checkpoint) == done['checkpoint_sha256'][str(step)]
            model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
                atlas_conditioning=True, fit_quality=True, vector_refinement=True,
                candidate_ranking=True, fitted_ranking=True).cuda()
            attach_global_plane_matcher(model, enabled=True)
            model.global_plane_matcher_120['evidence'] = nn.Sequential(
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)).cuda()
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert saved['step'] == step and saved['calibrated'] is False
            model.load_state_dict(saved['model'], strict=True)
            model.eval().requires_grad_(False)
            predictions, acquisition = {}, {}
            for section, record in records.items():
                with np.load(panel / record['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    acquisition[section] = (torch.from_numpy(arrays['offsets_um'].copy()).cuda(),
                                            torch.from_numpy(arrays['weights'].copy()).cuda())
                predicted = model.predict(image)
                predictions[section] = {key: predicted[key].cpu() for key in
                    ('feature', 'state', 'log_mass', 'reflection_logit')}

            variants = ('full', 'atlas_zero', 'source_swap') if arm == 'full' else ('support',)
            variant_rows = {name: [] for name in variants}
            for row in prior_rows:
                section = row['section_id']
                original = row['arms']['treatment']
                branches = np.asarray(original['beam_branch_ids'])
                mode = torch.from_numpy((branches // 2)[None].copy()).cuda()
                reflection = torch.from_numpy((branches % 2)[None].copy()).cuda()
                prior = {key: value.cuda() for key, value in predictions[section].items()}
                offsets, weights = acquisition[section]
                pair = original['original_only_pair_sensitivity']
                for variant in variants:
                    prediction = dict(prior)
                    if variant == 'source_swap':
                        prediction['feature'] = predictions[row['swap_section_id']]['feature'].cuda()
                    matched = global_plane_evidence_130(model, prediction, mode, reflection,
                        offsets, weights, atlas, (256, 256),
                        support_only=(variant in ('support', 'atlas_zero')))
                    score = matched['original_score'][0].cpu().numpy()
                    base = matched['input_score'][0].cpu().numpy()
                    assert np.max(np.abs(base - original['input_scores'])) < 1e-3
                    selected = int(score.argmax())
                    beam_prior = int(base.argmax())
                    mapped = original['original_mapped_mm_by_branch_id']
                    pair_win = (bool(score[pair['best_slot']] > score[pair['wrong_slot']])
                        if pair['status'] == 'scored' else None)
                    result = {'arm': arm, 'step': step, 'variant': variant,
                        'section_id': section, 'plan_id': row['synthetic_subject_plan_id'],
                        'appearance_mode': row['appearance_mode'],
                        'angle_deg': row['nearest_cardinal_angle_deg'],
                        'selected_slot': selected, 'selected_branch_id': int(branches[selected]),
                        'selected_original_mapped_mm': mapped[int(branches[selected])],
                        'beam_prior_slot': beam_prior,
                        'beam_prior_original_mapped_mm': mapped[int(branches[beam_prior])],
                        'global_prior_original_mapped_mm': mapped[int(
                            original['prior_ranked_branch_ids'][0])],
                        'best_blind_original_mapped_mm': min(mapped[int(b)] for b in branches),
                        'selected_changed_from_beam_prior': selected != beam_prior,
                        'pair_win': pair_win,
                        'pair_margin': (float(score[pair['best_slot']] - score[pair['wrong_slot']])
                            if pair_win is not None else None)}
                    variant_rows[variant].append(result)
                    all_rows.append(result)

            for variant, items in variant_rows.items():
                by_plan = defaultdict(list)
                for item in items:
                    by_plan[item['plan_id']].append(item)
                stats = {'sections': len(items), 'plans': len(by_plan),
                    'plan_equal_selected_mapped_mm': float(np.mean([
                        np.mean([x['selected_original_mapped_mm'] for x in plan])
                        for plan in by_plan.values()])),
                    'plan_equal_beam_prior_mapped_mm': float(np.mean([
                        np.mean([x['beam_prior_original_mapped_mm'] for x in plan])
                        for plan in by_plan.values()])),
                    'plan_equal_global_prior_mapped_mm': float(np.mean([
                        np.mean([x['global_prior_original_mapped_mm'] for x in plan])
                        for plan in by_plan.values()])),
                    'plan_equal_selected_le_1p5': float(np.mean([
                        np.mean([x['selected_original_mapped_mm'] <= 1.5 for x in plan])
                        for plan in by_plan.values()])),
                    'plan_equal_beam_prior_le_1p5': float(np.mean([
                        np.mean([x['beam_prior_original_mapped_mm'] <= 1.5 for x in plan])
                        for plan in by_plan.values()])),
                    'section_selected_le_1p5': float(np.mean([
                        x['selected_original_mapped_mm'] <= 1.5 for x in items])),
                    'selected_changed_from_beam_prior': sum(x['selected_changed_from_beam_prior']
                        for x in items),
                    'pair_sections': sum(x['pair_win'] is not None for x in items),
                    'pair_win_fraction': float(np.mean([x['pair_win'] for x in items
                        if x['pair_win'] is not None]))}
                summary[f'{arm}_{step}_{variant}'] = stats
                print(json.dumps({'event': 'eval_milestone', 'key': f'{arm}_{step}_{variant}',
                    **stats}), flush=True)
            del model, predictions

(output / 'synthetic_rows.jsonl').write_text('\n'.join(json.dumps(row, allow_nan=False)
    for row in all_rows) + '\n')
(output / 'summary.json').write_text(json.dumps(summary, indent=2))
(output / 'completed.json').write_text(json.dumps({
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'prior_eval_completion_sha256': sha(prior_eval / 'completed.json'),
    'source_sha256': sha(Path(__file__)),
    'summary_sha256': sha(output / 'summary.json'),
    'synthetic_rows_sha256': sha(output / 'synthetic_rows.jsonl'),
    'checkpoint_sha256': {arm: json.loads((root / f'runs/crossed_plane_evidence_130_{arm}'
        / 'completed.json').read_text())['checkpoint_sha256'] for arm in ('full', 'support')},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False}, indent=2))
