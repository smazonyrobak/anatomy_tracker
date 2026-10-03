"""Frozen 077 pose-correction readout on synthetic and weak-real development panels."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.normal_conditioned_cost_volume_077 import NormalConditionedCostVolume077

run = root / 'runs/normal_conditioned_cost_volume_077_pilot'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/normal_conditioned_cost_volume_077_development_eval'
steps, side, chunk = (0, 1000, 2000), 256, 2
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


def identity_mean(group, metric, identity):
    return float(np.mean([np.mean([row[metric] for row in group if row[identity] == key])
                          for key in {row[identity] for row in group}]))


training = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
assert finished['batches'] == 2000 and finished['accepted_synthetic'] == 4000
assert training['parent_sha256'] == sha(training['parent'])
baseline_path = root / 'runs/one_shot_dense_aux_075_development_eval/rows.jsonl'
baseline = {(row['set'], row['section_id']): row for row in
            map(json.loads, baseline_path.open()) if row['step'] == 0}
assert len(baseline) == 310
synthetic = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic) == 246 and len({row['synthetic_subject_plan_id'] for row in synthetic}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
support_cutoff = float(np.quantile([row['valid_pixels'] / side**2 for row in synthetic], .25))
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
frozen = torch.load(training['parent'], map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(frozen['model'], strict=True)
model.requires_grad_(False)
del frozen
head = NormalConditionedCostVolume077().cuda().eval()
head.requires_grad_(False)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def correct(image, offsets, weights):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = torch.cat((prior[:, :32].topk(8, -1).indices,
        prior[:, 32:].topk(6, -1).indices + 32), -1)
    state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    first, second = [], []
    for start in range(0, 14, chunk):
        a, b = head(prediction, image, state[:, start:start + chunk],
            choice[:, start:start + chunk] % 2, atlas, offsets, weights)
        first.append(a)
        second.append(b)
    return prior, choice, state, torch.cat(first, 1), torch.cat(second, 1)


out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'steps': steps, 'source_sha256': sha(Path(__file__)),
    'updater_source_sha256': sha(Path(__file__).parent / 'normal_conditioned_cost_volume_077.py'),
    'protocol_sha256': sha(Path(__file__).parent.parent /
        'docs/publication/NORMAL_CONDITIONED_COST_VOLUME_077_PROTOCOL_20261004.md'),
    'atlas_binding_source_sha256': sha(Path(__file__).parent / 'arbitrary_plane_allen_atlas_binding_v6.py'),
    'training_config_sha256': sha(run / 'config.json'),
    'training_completed_sha256': sha(run / 'completed.json'),
    'parent_sha256': training['parent_sha256'],
    'matched_059_parent_baseline': str(baseline_path),
    'matched_059_parent_baseline_sha256': sha(baseline_path),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt') for step in steps},
    'synthetic_records_sha256': sha(panel / 'records.jsonl'),
    'synthetic_panel_receipt_sha256': sha(panel / 'completed.json'),
    'real_records_sha256': sha(real / 'records.jsonl'),
    'real_images_sha256': sha(real / 'images.npy'),
    'real_geometry_sha256': sha(real / 'geometry.npz'),
    'synthetic_section_points': 1024, 'low_support_cutoff': support_cutoff,
    'selected_branch': 'unchanged 059 prior rank among actual 14 proposals',
    'best14_metric': 'truth-best corrected physical error; unavailable oracle diagnostic',
    'real_reference': 'inherited weak Allen affine, not expert ground truth',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        frozen = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert frozen['step'] == step and not frozen['calibrated']
        head.load_state_dict(frozen['head'], strict=True)
        del frozen
        for split, records in (('synthetic', synthetic), ('real_weak_allen', real_records)):
            for record in records:
                if split == 'synthetic':
                    path = panel / record['file']
                    assert sha(path) == record['sha256']
                    with np.load(path, allow_pickle=False) as arrays:
                        image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                        target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                        valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                        offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                        weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                    indices = valid.flatten().nonzero().flatten()
                    chosen = indices[torch.linspace(0, len(indices) - 1, 1024,
                                                    device='cuda').round().long()]
                    chart = torch.stack((chosen.remainder(side),
                        chosen.div(side, rounding_mode='floor')), -1).float() / side
                    reference = target.reshape(-1, 3)[chosen]
                else:
                    index = record['array_row_index']
                    image = np.concatenate((real_images[index].astype(np.float32),
                                            np.zeros((4, 192, 192), np.float32)))[None]
                    image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                          mode='bilinear', align_corners=False)
                    offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[index])
                    weights = torch.ones_like(offsets)
                    weights[:, [0, -1]] = .5
                    weights /= weights.sum(-1, keepdim=True)
                    chart = corners
                    affine = torch.as_tensor(affines[index], device='cuda', dtype=torch.float32)
                    reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                                + 192 * corners[:, 1:] * affine[:, 1]
                prior, choice, state, first, corrected = correct(image, offsets, weights)
                reflection = choice % 2
                original_error = (points(state, reflection, chart) - reference).norm(dim=-1).mean(-1)[0]
                first_error = (points(first, reflection, chart) - reference).norm(dim=-1).mean(-1)[0]
                corrected_error = (points(corrected, reflection, chart) - reference).norm(dim=-1).mean(-1)[0]
                if step == 0:
                    assert float((first_error - original_error).abs().max()) < 1
                    assert float((corrected_error - original_error).abs().max()) < 1
                selected = int(prior.gather(1, choice)[0].argmax())
                if step == 0:
                    assert abs(float(original_error[selected]) -
                        baseline[(split, record['section_id'])]['prior_selected_um']) < 1
                row = {'set': split, 'step': step, 'section_id': record['section_id'],
                    'parent_selected_um': float(original_error[selected]),
                    'parent_best14_oracle_um': float(original_error.min()),
                    'first_selected_um': float(first_error[selected]),
                    'first_best14_oracle_um': float(first_error.min()),
                    'corrected_selected_um': float(corrected_error[selected]),
                    'corrected_best14_oracle_um': float(corrected_error.min()),
                    'parent_selected_branch': int(choice[0, selected])}
                if split == 'synthetic':
                    row.update({key: record[key] for key in ('animal_id', 'synthetic_animal_id',
                        'specimen_id', 'experiment_id', 'synthetic_subject_plan_id',
                        'panel_physical_section_id', 'sha256', 'appearance_mode')})
                    row['visible_fraction'] = record['valid_pixels'] / side**2
                    row['low_support'] = row['visible_fraction'] <= support_cutoff
                    row['parent_near_true'] = row['parent_best14_oracle_um'] <= 1000
                else:
                    row.update({key: record[key] for key in ('animal_id', 'animal_partition_key',
                        'specimen_id', 'experiment_id', 'source_image_sha256',
                        'source_geometry_record_sha256')})
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
            stream.flush()
            print(json.dumps({'step': step, 'set': split, 'sections': len(records)}), flush=True)

metrics = ('parent_selected_um', 'parent_best14_oracle_um',
           'first_selected_um', 'first_best14_oracle_um',
           'corrected_selected_um', 'corrected_best14_oracle_um')
summary = []
for step in steps:
    entry = {'step': step}
    for split, identity in (('synthetic', 'synthetic_subject_plan_id'),
                            ('real_weak_allen', 'animal_id')):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        entry[split] = {'sections': len(group), 'identities': len({row[identity] for row in group}),
            'identity_equal_mean_um': {metric: identity_mean(group, metric, identity)
                                       for metric in metrics},
            'by_identity': {str(key): {metric: float(np.mean([row[metric] for row in group
                if row[identity] == key])) for metric in metrics}
                for key in sorted({row[identity] for row in group})}}
        if split == 'synthetic':
            entry[split]['by_appearance'] = {name: {metric: identity_mean(subset, metric, identity)
                for metric in metrics} for name in sorted({row['appearance_mode'] for row in group})
                if (subset := [row for row in group if row['appearance_mode'] == name])}
            entry[split]['by_support'] = {str(low): {metric: identity_mean(subset, metric, identity)
                for metric in metrics} for low in (True, False)
                if (subset := [row for row in group if row['low_support'] == low])}
            near = [row for row in group if row['parent_near_true']]
            entry[split]['near_true'] = {'sections': len(near),
                'identities': len({row[identity] for row in near}),
                'identity_equal_mean_um': {metric: identity_mean(near, metric, identity)
                                           for metric in metrics}}
    if step == 2000:
        synthetic_result = entry['synthetic']
        real_result = entry['real_weak_allen']
        gate = {
            'selected_le_2400_um': synthetic_result['identity_equal_mean_um']['corrected_selected_um'] <= 2400,
            'best14_le_820_um': synthetic_result['identity_equal_mean_um']['corrected_best14_oracle_um'] <= 820,
            'real_donors_le_parent_plus_200_um': all(
                donor['corrected_selected_um'] <= donor['parent_selected_um'] + 200
                for donor in real_result['by_identity'].values()),
            'appearance_le_parent_plus_200_um': all(
                stratum['corrected_selected_um'] <= stratum['parent_selected_um'] + 200
                for stratum in synthetic_result['by_appearance'].values()),
            'support_le_parent_plus_200_um': all(
                stratum['corrected_selected_um'] <= stratum['parent_selected_um'] + 200
                for stratum in synthetic_result['by_support'].values()),
            'near_true_best14_le_parent_plus_100_um':
                synthetic_result['near_true']['identity_equal_mean_um']['corrected_best14_oracle_um']
                <= synthetic_result['near_true']['identity_equal_mean_um']['parent_best14_oracle_um'] + 100,
        }
        gate['pass'] = all(gate.values())
        entry['gate'] = gate
    summary.append(entry)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
