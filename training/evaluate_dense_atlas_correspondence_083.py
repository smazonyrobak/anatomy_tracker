"""Frozen synthetic-DEV readout of directly supervised 083 atlas matches."""
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
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083, synthetic_match_targets

panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
pilot = root / 'runs/dense_atlas_correspondence_083_pilot'
out = root / 'runs/dense_atlas_correspondence_083_development_eval'
parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / 256 - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


receipt = json.loads((pilot / 'completed.json').read_text())
config = json.loads((pilot / 'config.json').read_text())
assert receipt['batches'] == 6000 and sha(pilot / 'config.json') == receipt['config_sha256']
assert sha(parent) == config['parent_sha256']
assert sha(pilot / 'draws.jsonl') == receipt['draws_sha256']
assert sha(pilot / 'training.jsonl') == receipt['training_sha256']
for name, digest in config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == digest
for step, digest in receipt['checkpoint_sha256'].items():
    assert sha(pilot / f'match_step_{int(step):05d}.pt') == digest
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 246 and len({row['synthetic_subject_plan_id'] for row in records}) == 8
out.mkdir(parents=True, exist_ok=False)
eval_config = {'pilot': str(pilot), 'pilot_completed_sha256': sha(pilot / 'completed.json'),
               'panel_records_sha256': sha(panel / 'records.jsonl'),
               'sections': len(records), 'synthetic_plans': 8,
               'checkpoints': config['checkpoints'], 'support_threshold': .5,
               'candidate_roles': ['physical_best_existing', 'uniform_other_fixed'],
               'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(eval_config, indent=2))
frozen = torch.load(parent, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(frozen['model'], strict=True)
model.requires_grad_(False)
del frozen
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in config['checkpoints']:
        checkpoint = torch.load(pilot / f'match_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        head = WholeSliceAtlasFeedback083().cuda().eval()
        head.load_state_dict(checkpoint['head'], strict=True)
        head.requires_grad_(False)
        del checkpoint
        for number, record in enumerate(records):
            if step == 0:
                assert sha(panel / record['file']) == record['sha256']
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                true_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda().float()
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            visible = valid.flatten().nonzero()[:, 0]
            pixel = visible[torch.linspace(0, len(visible) - 1, 96, device='cuda').long()][None]
            sample = target.reshape(1, -1, 3).gather(1, pixel[..., None].expand(-1, -1, 3))
            chart = torch.stack((pixel.remainder(256),
                                 pixel.div(256, rounding_mode='floor')), -1).float() / 256
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = torch.arange(2, device='cuda')[None, None].expand(1, model.modes, 2)
            distance = (points(states, reflections, chart) - sample[:, None, None]).norm(dim=-1).mean(-1)
            true_normal = full_frame_state_to_components(true_state)[1][..., :, 2]
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            distance += 4000 * (1 - (normals * true_normal[:, None]).sum(-1).abs().clamp_max(1))[..., None]
            best = distance.flatten(1).gather(1, beam).argmin(-1)
            other = torch.tensor([(number * 7 + 3) % 13], device='cuda')
            other += (other >= best).long()
            choice = beam.gather(1, torch.stack((best, other), -1))
            state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            reflection = choice % 2
            labels = synthetic_match_targets(target, valid, state, reflection)
            output = head(prediction['feature'], state, reflection, atlas, offsets,
                          weights, match_only=True)
            centre, frame, basis = full_frame_state_to_components(state)
            edges = frame[..., :, :2] @ basis
            for scale, side, radius in (('fine', 32, 2), ('coarse', 16, 4)):
                logits = output[f'{scale}_match_logits']
                index = labels[f'{scale}_index']
                support = output[f'{scale}_match_support'].gather(2, index[:, :, None]).squeeze(2)
                mask = labels[f'{scale}_mask'] & (support >= .5)
                ce = F.cross_entropy(logits.flatten(0, 1), index.flatten(0, 1),
                                     reduction='none').reshape_as(mask)
                predicted = logits.argmax(2)
                block = (2 * radius + 1) ** 2
                z = predicted.div(block, rounding_mode='floor') * 1500 - 6000
                lateral = predicted.remainder(block)
                dy = lateral.div(2 * radius + 1, rounding_mode='floor') - radius
                dx = lateral.remainder(2 * radius + 1) - radius
                y = (torch.arange(side, device='cuda').float() + .5) / side - .5 / 256
                x = (torch.arange(side, device='cuda').float() + .5) / side - .5 / 256
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                sx = torch.where(reflection[..., None, None].bool(), 255 / 256 - xx, xx)
                sx = sx + torch.where(reflection[..., None, None].bool(), -1., 1.) * dx / side
                xy = torch.stack((sx, yy + dy / side), -1)
                world = (centre[..., None, None, :] + torch.einsum(
                    'bkij,bkhwj->bkhwi', edges, xy - .5)
                    + z[..., None] * frame[..., None, None, :, 2])
                truth = F.interpolate(target.permute(0, 3, 1, 2), (side, side),
                                      mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                error = (world - truth[:, None]).norm(dim=-1) / 1000
                for slot, role in enumerate(('physical_best_existing', 'uniform_other_fixed')):
                    count = int(mask[0, slot].sum())
                    row = {'step': step, 'section_id': record['section_id'],
                           'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                           'appearance_mode': record['appearance_mode'],
                           'role': role, 'branch': int(choice[0, slot]), 'scale': scale,
                           'geometric_centres': int(labels[f'{scale}_mask'][0, slot].sum()),
                           'supported_centres': count,
                           'ce': float(ce[0, slot][mask[0, slot]].mean()) if count else None,
                           'top1_mm': float(error[0, slot][mask[0, slot]].mean()) if count else None}
                    rows.append(row)
                    stream.write(json.dumps(row) + '\n')
        print(json.dumps({'checkpoint': step, 'rows': len(rows)}), flush=True)

summary = []
for step in config['checkpoints']:
    for scale in ('fine', 'coarse'):
        for role in ('physical_best_existing', 'uniform_other_fixed', 'all'):
            group = [row for row in rows if row['step'] == step and row['scale'] == scale
                     and (role == 'all' or row['role'] == role)]
            plans = {row['synthetic_subject_plan_id'] for row in group}
            usable = [row for row in group if row['supported_centres']]
            valid_plans = sorted({row['synthetic_subject_plan_id'] for row in usable})
            plan_ce = [np.mean([row['ce'] for row in usable if row['synthetic_subject_plan_id'] == plan])
                       for plan in valid_plans]
            plan_top1 = [np.mean([row['top1_mm'] for row in usable if row['synthetic_subject_plan_id'] == plan])
                          for plan in valid_plans]
            summary.append({'step': step, 'scale': scale, 'role': role,
                'sections': len({row['section_id'] for row in group}),
                'synthetic_plans': len(plans), 'supported_rows': len(usable),
                'supported_centres': sum(row['supported_centres'] for row in group),
                'geometric_centres': sum(row['geometric_centres'] for row in group),
                'plan_equal_ce': float(np.mean(plan_ce)),
                'plan_equal_top1_mm': float(np.mean(plan_top1))})
baseline = {(row['scale'], row['role']): row for row in summary if row['step'] == 0}
for row in summary:
    initial = baseline[(row['scale'], row['role'])]
    row['ce_change_fraction'] = 1 - row['plan_equal_ce'] / initial['plan_equal_ce']
    row['top1_change_fraction'] = 1 - row['plan_equal_top1_mm'] / initial['plan_equal_top1_mm']
final = {(row['scale'], row['role']): row for row in summary if row['step'] == 6000}
gate = {'coarse_ce_gain_ge_20_percent': final[('coarse', 'all')]['ce_change_fraction'] >= .2,
        'coarse_top1_gain_ge_20_percent': final[('coarse', 'all')]['top1_change_fraction'] >= .2,
        'fine_ce_regression_le_5_percent': final[('fine', 'all')]['ce_change_fraction'] >= -.05}
gate['pass'] = all(gate.values())
(out / 'summary.json').write_text(json.dumps({'metrics': summary, 'gate': gate}, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
