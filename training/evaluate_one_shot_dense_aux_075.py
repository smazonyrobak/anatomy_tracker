"""Frozen direct-pose and dense-plane readout on the independent 061 development panel."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_dense_aux_075'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_dense_aux_075_development_eval'
steps, side = (0, 2000, 5000, 10000), 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], len(chart), 2).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


def identity_mean(rows, field, key):
    identities = {row[key] for row in rows}
    return float(np.mean([np.mean([row[field] for row in rows if row[key] == identity])
                          for identity in identities]))


training = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
assert finished['batches'] == 10000 and finished['accepted_synthetic'] == 20000
assert training['parent_sha256'] == sha(training['parent'])
synthetic = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic) == 246 and len({row['synthetic_subject_plan_id'] for row in synthetic}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
support_cutoff = float(np.quantile([row['valid_pixels'] / side**2 for row in synthetic], .25))
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True,
    dense_coordinate_pyramid=True).cuda().eval()
model.requires_grad_(False)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'steps': steps, 'source_sha256': sha(Path(__file__)),
    'training_config_sha256': sha(run / 'config.json'),
    'training_completed_sha256': sha(run / 'completed.json'),
    'parent_sha256': training['parent_sha256'],
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt') for step in steps},
    'synthetic_records_sha256': sha(panel / 'records.jsonl'),
    'synthetic_panel_receipt_sha256': sha(panel / 'completed.json'),
    'real_records_sha256': sha(real / 'records.jsonl'),
    'real_images_sha256': sha(real / 'images.npy'),
    'real_geometry_sha256': sha(real / 'geometry.npz'),
    'synthetic_section_points': 1024,
    'synthetic_raw_field_metric': 'bilinear 128-to-256 CCF field; mean Euclidean error at surviving tissue pixels',
    'synthetic_validity_metric': '128-grid Dice, predicted logit >= 0 versus pooled valid fraction >= 0.5',
    'low_support_definition': 'bottom quartile of eligible 061 DEV visible-tissue fraction',
    'low_support_cutoff': support_cutoff,
    'synthetic_identities': 8, 'real_development_donors': 6,
    'synthetic_reference': 'known tissue-to-CCF coordinates, not biological animal truth',
    'real_reference': 'weak inherited Allen affine at five points, not expert truth',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        frozen = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert frozen['step'] == step and not frozen['calibrated']
        model.load_state_dict(frozen['model'], strict=True)
        del frozen
        for record in synthetic:
            path = panel / record['file']
            assert sha(path) == record['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            indices = valid.flatten().nonzero().flatten()
            chosen = indices[torch.linspace(0, len(indices) - 1, 1024, device='cuda').round().long()]
            chart = torch.stack((chosen.remainder(side), chosen.div(side, rounding_mode='floor')),
                                -1).float() / side
            reference = target.reshape(-1, 3)[chosen]
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = torch.cat((prior[:, :32].topk(2, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            states = prediction['state'].repeat_interleave(2, 1)
            flags = torch.arange(2, device='cuda').repeat(model.modes)[None]
            errors = (points(states, flags, chart) - reference).norm(dim=-1).mean(-1)[0]
            dense_state = model.dense_coordinate_plane(prediction)[:, None]
            dense_error = (points(dense_state, torch.zeros((1, 1), device='cuda'), chart)
                           - reference).norm(dim=-1).mean()
            field = prediction['dense_coordinate']
            raw = F.interpolate(field[:, :3], (side, side), mode='bilinear', align_corners=False)[0]
            raw = raw.permute(1, 2, 0) * model.center_scale + model.center_origin
            raw_tissue_error = (raw[valid] - target[valid]).norm(dim=-1).mean()
            valid_128 = F.avg_pool2d(valid[None, None].float(), 2)[0, 0] >= .5
            predicted_valid_128 = field[0, 3] >= 0
            valid_dice = (2 * (predicted_valid_128 & valid_128).sum()
                          / (predicted_valid_128.sum() + valid_128.sum()).clamp_min(1))
            selected = int(prior[0].argmax())
            beam_error = errors[choice[0]]
            row = {'set': 'synthetic', 'step': step,
                **{key: record[key] for key in ('animal_id', 'synthetic_animal_id',
                    'specimen_id', 'experiment_id', 'section_id', 'synthetic_subject_plan_id',
                    'panel_physical_section_id', 'sha256', 'appearance_mode')},
                'visible_fraction': record['valid_pixels'] / side**2,
                'low_support': record['valid_pixels'] / side**2 <= support_cutoff,
                'prior_selected_um': float(errors[selected]),
                'prior_best8_um': float(beam_error.min()),
                'prior_best160_um': float(errors.min()),
                'field_fit_um': float(dense_error),
                'field_raw_tissue_um': float(raw_tissue_error),
                'field_valid_dice': float(valid_dice),
                'field_validity_mean': float(prediction['dense_coordinate'][:, 3].sigmoid().mean()),
                'prior_selected_branch': selected}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real_records:
            index = record['array_row_index']
            image = np.concatenate((real_images[index].astype(np.float32),
                                    np.zeros((4, 192, 192), np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            affine = torch.as_tensor(affines[index], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                        + 192 * corners[:, 1:] * affine[:, 1]
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = torch.cat((prior[:, :32].topk(2, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            states = prediction['state'].repeat_interleave(2, 1)
            flags = torch.arange(2, device='cuda').repeat(model.modes)[None]
            errors = (points(states, flags, corners) - reference).norm(dim=-1).mean(-1)[0]
            dense_state = model.dense_coordinate_plane(prediction)[:, None]
            dense_error = (points(dense_state, torch.zeros((1, 1), device='cuda'), corners)
                           - reference).norm(dim=-1).mean()
            row = {'set': 'real_weak_allen', 'step': step,
                **{key: record[key] for key in ('animal_id', 'animal_partition_key',
                    'specimen_id', 'experiment_id', 'section_id', 'source_image_sha256',
                    'source_geometry_record_sha256')},
                'prior_selected_um': float(errors[int(prior[0].argmax())]),
                'prior_best8_um': float(errors[choice[0]].min()),
                'prior_best160_um': float(errors.min()),
                'field_fit_um': float(dense_error),
                'field_validity_mean': float(prediction['dense_coordinate'][:, 3].sigmoid().mean())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic': len(synthetic),
                          'real_weak': len(real_records)}), flush=True)

metrics = ('prior_selected_um', 'prior_best8_um', 'prior_best160_um', 'field_fit_um')
summary = []
for step in steps:
    entry = {'step': step}
    for split, identity in (('synthetic', 'synthetic_subject_plan_id'),
                            ('real_weak_allen', 'animal_id')):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        scored_metrics = (metrics + ('field_raw_tissue_um',)
                          if split == 'synthetic' else metrics)
        reported_metrics = (scored_metrics + ('field_valid_dice',)
                            if split == 'synthetic' else scored_metrics)
        entry[split] = {'sections': len(group), 'identities': len({row[identity] for row in group}),
            'identity_equal_mean_um': {name: identity_mean(group, name, identity)
                                       for name in scored_metrics},
            'by_identity': {str(key): {name: float(np.mean([row[name] for row in group
                if row[identity] == key])) for name in reported_metrics}
                for key in sorted({row[identity] for row in group})}}
        if split == 'synthetic':
            entry[split]['identity_equal_mean_dice'] = identity_mean(
                group, 'field_valid_dice', identity)
            entry[split]['by_appearance'] = {appearance: {
                'sections': len(subset), 'identity_equal_mean_um': {
                    name: identity_mean(subset, name, identity) for name in scored_metrics},
                'identity_equal_mean_dice': identity_mean(subset, 'field_valid_dice', identity)}
                for appearance in sorted({row['appearance_mode'] for row in group})
                if (subset := [row for row in group if row['appearance_mode'] == appearance])}
            entry[split]['by_support'] = {str(low): {
                'sections': len(subset), 'identity_equal_mean_um': {
                    name: identity_mean(subset, name, identity) for name in scored_metrics},
                'identity_equal_mean_dice': identity_mean(subset, 'field_valid_dice', identity)}
                for low in (True, False)
                if (subset := [row for row in group if row['low_support'] == low])}
            entry[split]['selected_over_5mm'] = sum(
                row['prior_selected_um'] > 5000 for row in group)
            entry[split]['field_fit_over_5mm'] = sum(
                row['field_fit_um'] > 5000 for row in group)
    summary.append(entry)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
