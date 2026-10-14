"""Read-only 305 spatial-score replay on 64 frozen 309 v10 TRAIN-inner cases."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/first_choice_exposure_309_001'
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert Path(__file__).resolve().drive.upper() == RUN.drive.upper() == 'I:'
config = json.loads((RUN / 'config.json').read_text())
receipt = json.loads((RUN / 'completed.json').read_text())
arm_receipt = json.loads((RUN / 'single/completed.json').read_text())
assert config['candidate'] == receipt['candidate'] == 309 and receipt['selected_step'] == 20000
with (RUN / 'panels.pt').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == config['panels_sha256']
assert hashlib.sha256(Path('training/arbitrary_plane_one_shot_model.py').read_bytes()).hexdigest() == config['model_source_sha256']
assert hashlib.sha256(Path('training/arbitrary_plane_one_shot_inference.py').read_bytes()).hexdigest() == config['inference_source_sha256']
for step in (0, 20000):
    name = f'pose_local_{step:05d}.pt'
    with (RUN / 'single' / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == arm_receipt['output_sha256'][name]

panels = torch.load(RUN / 'panels.pt', map_location='cpu', weights_only=True)
panel = panels['v10'][:64]
assert len(panel) == 64 and [sum(item['meta']['base'] == base for item in panel)
                             for base in range(56, 64)] == [8] * 8
context = load_fov_balance_v10('cuda')
model0, _ = load_one_shot_checkpoint(RUN / 'single/pose_local_00000.pt')
model, _ = load_one_shot_checkpoint(RUN / 'single/pose_local_20000.pt')
model0.requires_grad_(False)
model.requires_grad_(False)
assert model.spatial_feedback and model.modes == 80
rows = []
with torch.inference_mode():
    for slot, item in enumerate(panel):
        meta = item['meta']
        sample = sample_fov_balance_v10(context,
            [meta['base'] * VARIANTS + meta['variant']], meta['seed'], 192)
        assert sample['provenance'][0]['physical_section_id'] == meta['physical_section_id']
        assert torch.equal(sample['inputs'][0].cpu(), item['inputs'])
        assert torch.equal(sample['state'][0].cpu(), item['state'])
        pixels = (item['chart'][:, 1] * 192).round().long() * 192 + (
            item['chart'][:, 0] * 192).round().long()
        assert bool(sample['valid_mask'][0].flatten()[pixels].all())
        assert torch.equal(sample['centre'][0].reshape(-1, 3)[pixels].cpu(), item['truth'])
        inputs = item['inputs'][None].cuda()
        chart = item['chart'][None].cuda()
        truth = item['truth'][None].cuda()
        offsets, weights = sample['offsets'], sample['weights']
        prediction = model.predict(inputs)
        q_rows = {}
        for name, evaluator in (('step0', model0), ('step20000', model)):
            current = evaluator.predict(inputs) if name == 'step0' else prediction
            q = current['reliability_logit'].sigmoid()
            valid = F.interpolate(sample['valid_mask'][:, None].float(),
                                  q.shape[-2:], mode='area')
            q_rows[name] = {'valid_mean': float((q * valid).sum() / valid.sum()),
                'background_mean': float((q * (1 - valid)).sum() / (1 - valid).sum()),
                'valid_above_half': float(((q > .5) * valid).sum() / valid.sum()),
                'background_above_half': float(((q > .5) * (1 - valid)).sum() /
                                               (1 - valid).sum())}
            if slot < 16:
                true_prediction = {**current, 'state': sample['state'][:, None]}
                true_map = evaluator.map(true_prediction, offsets, torch.zeros((1, 1),
                    device='cuda', dtype=torch.long), sample['reflection'][:, None],
                    (96, 96), context['atlas'], weights, feature_side=96,
                    source_shape=(192, 192), max_derivative=None)
                surface = true_map['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1)
                grid = ((chart + .5 / 192) * 2 - 1).reshape(1, 1, 128, 2)
                points = F.grid_sample(surface.permute(0, 3, 1, 2), grid,
                    padding_mode='border', align_corners=False)[:, :, 0].transpose(1, 2)
                q_rows[name]['true_pose_map_error_mm'] = float(
                    (points - truth).norm(dim=-1).mean() / 1000)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branches = prior.topk(16, -1).indices
        scores = {'full': [], 'support_only': [], 'shuffled': []}
        rigid_errors, mapped_errors, support_means = [], [], []
        for start in range(0, 16, 2):
            group = branches[:, start:start + 2]
            modes = group // 2
            selected = {**prediction,
                'state': prediction['state'].gather(1, modes[..., None].expand(-1, -1, 12)),
                'log_mass': prediction['log_mass'].gather(1, modes),
                'reflection_logit': prediction['reflection_logit'].gather(1, modes)}
            index = torch.arange(2, device='cuda')[None]
            rigid = rigid_points_090(selected['state'], group % 2,
                                     chart[:, None], (192, 192))
            rigid_errors.extend(((rigid - truth[:, None]).norm(dim=-1).mean(-1) /
                                 1000)[0].tolist())
            first = model.map(selected, offsets, index, group % 2, (64, 64),
                context['atlas'], weights, return_refinement_feature=True,
                feature_side=64, source_shape=(192, 192), max_derivative=None)
            refined_state, _, _ = model.refine(first['refinement_feature'], selected['state'])
            refined = {**selected, 'state': refined_state}
            fitted = model.map(refined, offsets, index, group % 2, (96, 96),
                context['atlas'], weights, return_refinement_feature=True,
                feature_side=96, source_shape=(192, 192), max_derivative=None)
            surface = fitted['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1)
            grid = ((chart + .5 / 192) * 2 - 1)[:, None].expand(-1, 2, -1, -1)
            grid = grid.reshape(2, 1, 128, 2)
            points = F.grid_sample(surface.permute(0, 3, 1, 2), grid,
                padding_mode='border', align_corners=False)[:, :, 0].transpose(1, 2)
            mapped_errors.extend(((points.reshape(1, 2, 128, 3) - truth[:, None])
                                  .norm(dim=-1).mean(-1) / 1000)[0].tolist())
            support_means.extend(fitted['atlas_pair'][0, :, 1].mean((-2, -1)).tolist())
            for arm in scores:
                scores[arm].extend(model.score_spatial_candidates(inputs, selected, fitted,
                    context['atlas'], weights, evidence_arm=arm)[0].tolist())
        score_prior = prior[0, branches[0]].tolist()
        chosen = {'prior': 0, **{arm: int(np.argmax(values)) for arm, values in scores.items()}}
        near = int(np.argmin(rigid_errors))
        far = [i for i in range(16) if rigid_errors[near] <= 1.5
            and rigid_errors[i] >= max(2., rigid_errors[near] + 1.)
            and abs(support_means[i] - support_means[near]) <= .10]
        far = max(far, key=lambda i: score_prior[i]) if far else None
        rows.append({'physical_section_id': meta['physical_section_id'], 'base': meta['base'],
            'slot': slot, 'q': q_rows, 'rigid_error_mm': rigid_errors,
            'mapped_error_mm': mapped_errors, 'support_mean': support_means,
            'prior_log': score_prior, 'scores': scores, 'chosen': chosen,
            'matched_pair': None if far is None else (near, far)})
        if (slot + 1) % 8 == 0:
            print(json.dumps({'replayed': slot + 1, 'of': len(panel)}), flush=True)

arms = ('prior', 'full', 'support_only', 'shuffled')
summary = {'cases': len(rows), 'bases': list(range(56, 64)),
    'checkpoint_step': 20000, 'panel': 'frozen 309 v10, first 64 slots',
    'source_reliability': {name: {metric: float(np.mean([r['q'][name][metric]
        for r in rows])) for metric in ('valid_mean', 'background_mean',
            'valid_above_half', 'background_above_half')}
        for name in ('step0', 'step20000')},
    'true_pose_map_first16_mm': {name: float(np.mean([
        r['q'][name]['true_pose_map_error_mm'] for r in rows[:16]]))
        for name in ('step0', 'step20000')},
    'matched_pairs': sum(r['matched_pair'] is not None for r in rows)}
for geometry in ('rigid', 'mapped'):
    error = geometry + '_error_mm'
    summary[geometry] = {arm: {'first_mean_mm': float(np.mean([
        r[error][r['chosen'][arm]] for r in rows])),
        'first_minus_top16_oracle_mean_mm': float(np.mean([
            r[error][r['chosen'][arm]] - min(r[error]) for r in rows])),
        'first_over5mm': sum(r[error][r['chosen'][arm]] > 5 for r in rows),
        'new_over5mm_vs_prior': sum(r[error][r['chosen'][arm]] > 5 >=
            r[error][r['chosen']['prior']] for r in rows)} for arm in arms}
    summary[geometry]['top16_oracle_mean_mm'] = float(np.mean([min(r[error]) for r in rows]))
summary['full_incremental_rigid_vs_controls'] = {control: {
    'mean_error_gain_mm': float(np.mean([r['rigid_error_mm'][r['chosen'][control]] -
        r['rigid_error_mm'][r['chosen']['full']] for r in rows])),
    'cases_full_better': sum(r['rigid_error_mm'][r['chosen']['full']] <
        r['rigid_error_mm'][r['chosen'][control]] - 1e-9 for r in rows),
    'cases_full_worse': sum(r['rigid_error_mm'][r['chosen']['full']] >
        r['rigid_error_mm'][r['chosen'][control]] + 1e-9 for r in rows),
    'choice_changed': sum(r['chosen']['full'] != r['chosen'][control] for r in rows)}
    for control in ('prior', 'support_only', 'shuffled')}
summary['matched_pair_evidence_wins'] = {arm: sum(
    r['scores'][arm][r['matched_pair'][0]] - r['prior_log'][r['matched_pair'][0]] >
    r['scores'][arm][r['matched_pair'][1]] - r['prior_log'][r['matched_pair'][1]]
    for r in rows if r['matched_pair'] is not None)
    for arm in ('full', 'support_only', 'shuffled')}
summary['rigid_first_mean_by_base'] = {str(base): {arm: float(np.mean([
    r['rigid_error_mm'][r['chosen'][arm]] for r in rows if r['base'] == base]))
    for arm in arms} for base in range(56, 64)}
print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
