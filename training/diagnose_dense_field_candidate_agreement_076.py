"""Frozen 075 dense-field agreement ranking of its actual 160 pose candidates."""
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
previous = root / 'runs/one_shot_dense_aux_075_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/dense_field_candidate_agreement_076_development_diagnostic'
checkpoint = run / 'joint_step_10000.pt'
side, field_side = 256, 128
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


def identity_mean(group, metric, identity):
    return float(np.mean([np.mean([row[metric] for row in group if row[identity] == key])
                          for key in {row[identity] for row in group}]))


training = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
previous_config = json.loads((previous / 'config.json').read_text())
previous_finished = json.loads((previous / 'completed.json').read_text())
checkpoint_sha = sha(checkpoint)
assert finished['batches'] == 10000 and finished['accepted_synthetic'] == 20000
assert training['parent_sha256'] == sha(training['parent'])
assert previous_finished['config_sha256'] == sha(previous / 'config.json')
assert previous_finished['rows_sha256'] == sha(previous / 'rows.jsonl')
assert previous_config['parent_sha256'] == training['parent_sha256']
assert previous_config['checkpoint_sha256']['10000'] == checkpoint_sha
assert previous_config['checkpoint_sha256']['0'] == sha(run / 'joint_step_00000.pt')
previous_rows = [json.loads(line) for line in (previous / 'rows.jsonl').open()]
baseline = {(row['set'], row['section_id']): row for row in previous_rows if row['step'] == 0}
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

frozen = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert frozen['step'] == 10000 and not frozen['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True,
    dense_coordinate_pyramid=True).cuda().eval()
model.load_state_dict(frozen['model'], strict=True)
model.requires_grad_(False)
del frozen
axis = (torch.arange(0, field_side, 4, device='cuda').float() + .5) / field_side - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
field_chart = torch.stack((xx, yy), -1).reshape(-1, 2)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.arange(2, device='cuda').repeat(model.modes)[None]

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'source_sha256': sha(Path(__file__)),
    'protocol_sha256': sha(Path(__file__).parent.parent / 'docs/publication/DENSE_FIELD_CANDIDATE_AGREEMENT_076_PROTOCOL_20261004.md'),
    'training_config_sha256': sha(run / 'config.json'),
    'training_completed_sha256': sha(run / 'completed.json'),
    'checkpoint_10000_sha256': checkpoint_sha,
    'parent_059_sha256': training['parent_sha256'],
    'previous_eval_config_sha256': sha(previous / 'config.json'),
    'previous_eval_completed_sha256': sha(previous / 'completed.json'),
    'previous_eval_rows_sha256': previous_finished['rows_sha256'],
    'synthetic_records_sha256': sha(panel / 'records.jsonl'),
    'synthetic_panel_receipt_sha256': sha(panel / 'completed.json'),
    'real_records_sha256': sha(real / 'records.jsonl'),
    'real_images_sha256': sha(real / 'images.npy'),
    'real_geometry_sha256': sha(real / 'geometry.npz'),
    'score': 'predicted-validity-weighted lowest 80% of Huber 3D CCF distances, bend 1000 um',
    'score_grid_source': [field_side, field_side],
    'score_grid_sampled': [field_side // 4, field_side // 4],
    'score_stride': 4,
    'score_sample_offset': [0, 0],
    'top8_metric': 'truth-best physical error among the eight field-score-ranked candidates; oracle availability, not selected accuracy',
    'synthetic_metric': '075 evaluator: 1024 evenly spaced surviving-tissue pixels, mean Euclidean CCF error',
    'real_metric': '075 evaluator: five-point disagreement with inherited weak Allen affine',
    'low_support_cutoff': support_cutoff,
    'baseline': '059-derived 075 step-0 rows from the frozen prior evaluation',
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for split, records in (('synthetic', synthetic), ('real_weak_allen', real_records)):
        for record in records:
            old = baseline[(split, record['section_id'])]
            if split == 'synthetic':
                assert old['sha256'] == record['sha256']
                path = panel / record['file']
                assert sha(path) == record['sha256']
                with np.load(path, allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            else:
                assert old['source_image_sha256'] == record['source_image_sha256']
                index = record['array_row_index']
                image = np.concatenate((real_images[index].astype(np.float32),
                                        np.zeros((4, 192, 192), np.float32)))[None]
                image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                      mode='bilinear', align_corners=False)

            prediction = model.predict(image)
            states = prediction['state'].repeat_interleave(2, 1)
            assert states.shape[1] == 160
            full_field = prediction['dense_coordinate']
            assert full_field.shape[-2:] == (field_side, field_side)
            field = full_field[:, :, ::4, ::4]
            field_ccf = (field[:, :3] * model.center_scale[None, :, None, None]
                         + model.center_origin[None, :, None, None]).flatten(2).transpose(1, 2)
            distance = (points(states, flags, field_chart) - field_ccf[:, None]).norm(dim=-1)[0]
            huber = torch.where(distance <= 1000, distance.square() / 2000, distance - 500)
            ordered, indices = huber.sort(dim=-1)
            weight = field[:, 3].sigmoid().flatten(1).expand(160, -1).gather(1, indices)
            cutoff = .8 * weight[0].sum()
            retained = torch.minimum(weight, (cutoff - weight.cumsum(-1) + weight).clamp_min(0))
            score = (ordered * retained).sum(-1) / cutoff
            selected = int(score.argmin())
            top8 = score.topk(8, largest=False).indices

            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)[0]
            prior_selected = int(prior.argmax())
            prior8 = torch.cat((prior[:32].topk(2).indices,
                                prior[32:].topk(6).indices + 32))
            if split == 'synthetic':
                with np.load(path, allow_pickle=False) as arrays:
                    target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                chosen = valid.flatten().nonzero().flatten()
                chosen = chosen[torch.linspace(0, len(chosen) - 1, 1024,
                                               device='cuda').round().long()]
                chart = torch.stack((chosen.remainder(side),
                    chosen.div(side, rounding_mode='floor')), -1).float() / side
                reference = target.reshape(-1, 3)[chosen]
            else:
                chart = corners
                affine = torch.as_tensor(affines[index], device='cuda', dtype=torch.float32)
                reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                            + 192 * corners[:, 1:] * affine[:, 1]
            errors = (points(states, flags, chart) - reference).norm(dim=-1).mean(-1)[0]
            row = {'set': split, 'section_id': record['section_id'],
                'parent_059_selected_um': old['prior_selected_um'],
                'parent_059_best8_oracle_um': old['prior_best8_um'],
                'prior_selected_um': float(errors[prior_selected]),
                'prior_best8_oracle_um': float(errors[prior8].min()),
                'prior_best160_oracle_um': float(errors.min()),
                'field_selected_um': float(errors[selected]),
                'field_best8_oracle_um': float(errors[top8].min()),
                'field_selected_score_um': float(score[selected]),
                'prior_selected_score_um': float(score[prior_selected]),
                'field_selected_branch': selected,
                'field_top8_branches': top8.tolist(),
                'prior_selected_branch': prior_selected}
            if split == 'synthetic':
                row.update({key: record[key] for key in ('animal_id', 'synthetic_animal_id',
                    'specimen_id', 'experiment_id', 'synthetic_subject_plan_id',
                    'panel_physical_section_id', 'sha256', 'appearance_mode')})
                row['visible_fraction'] = record['valid_pixels'] / side**2
                row['low_support'] = row['visible_fraction'] <= support_cutoff
            else:
                row.update({key: record[key] for key in ('animal_id', 'animal_partition_key',
                    'specimen_id', 'experiment_id', 'source_image_sha256',
                    'source_geometry_record_sha256')})
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'set': split, 'sections': len(records)}), flush=True)

metrics = ('parent_059_selected_um', 'parent_059_best8_oracle_um', 'prior_selected_um',
           'prior_best8_oracle_um', 'prior_best160_oracle_um', 'field_selected_um',
           'field_best8_oracle_um')
summary = {}
for split, identity in (('synthetic', 'synthetic_subject_plan_id'),
                        ('real_weak_allen', 'animal_id')):
    group = [row for row in rows if row['set'] == split]
    summary[split] = {'sections': len(group), 'identities': len({row[identity] for row in group}),
        'identity_equal_mean_um': {metric: identity_mean(group, metric, identity)
                                   for metric in metrics},
        'by_identity': {str(key): {metric: float(np.mean([row[metric] for row in group
            if row[identity] == key])) for metric in metrics}
            for key in sorted({row[identity] for row in group})}}
    if split == 'synthetic':
        summary[split]['by_appearance'] = {name: {metric: identity_mean(subset, metric, identity)
            for metric in metrics} for name in sorted({row['appearance_mode'] for row in group})
            if (subset := [row for row in group if row['appearance_mode'] == name])}
        summary[split]['by_support'] = {str(low): {metric: identity_mean(subset, metric, identity)
            for metric in metrics} for low in (True, False)
            if (subset := [row for row in group if row['low_support'] == low])}

synthetic_result = summary['synthetic']
real_result = summary['real_weak_allen']
gate = {
    'synthetic_selected_le_2350_um': synthetic_result['identity_equal_mean_um']['field_selected_um'] <= 2350,
    'synthetic_best8_le_950_um': synthetic_result['identity_equal_mean_um']['field_best8_oracle_um'] <= 950,
    'real_donors_le_parent_plus_200_um': all(
        donor['field_selected_um'] <= donor['parent_059_selected_um'] + 200
        for donor in real_result['by_identity'].values()),
    'synthetic_appearance_le_parent_plus_200_um': all(
        stratum['field_selected_um'] <= stratum['parent_059_selected_um'] + 200
        for stratum in synthetic_result['by_appearance'].values()),
    'synthetic_support_le_parent_plus_200_um': all(
        stratum['field_selected_um'] <= stratum['parent_059_selected_um'] + 200
        for stratum in synthetic_result['by_support'].values()),
}
gate['pass'] = all(gate.values())
summary['gate'] = gate
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows), 'gate': gate}), flush=True)
