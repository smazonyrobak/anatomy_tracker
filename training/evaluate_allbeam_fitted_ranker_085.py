"""Independent full-beam 085 development readout against frozen 081 parent rows."""
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
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

run = root / 'runs/allbeam_fitted_ranker_085_pilot'
out = root / 'runs/allbeam_fitted_ranker_085_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
prior_eval = root / 'runs/whole_slice_joint_feedback_081_development_eval'
side = 256
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


def fit_summary(mapped):
    local = mapped['local_displacement_um'] / 1000
    magnitude = local.square().sum(2).sqrt().mean((-2, -1))
    roughness = ((local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
                 + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1)))
    support = mapped['atlas_pair'][:, :, 1].mean((-2, -1))
    reliability = mapped['correspondence_logit'].sigmoid().mean((-3, -2, -1))
    return torch.stack((magnitude, roughness, support, reliability), -1)


def candidate_prediction(prediction, choice, state):
    mode = choice // 2
    return {**prediction, 'state': state,
            'log_mass': prediction['log_mass'].gather(1, mode),
            'reflection_logit': prediction['reflection_logit'].gather(1, mode)}


def infer(image, offsets, weights, model, head, atlas):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                        prior[:, 32:].topk(6, -1).indices + 32), -1)
    state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    states, scores, surfaces, evidence = [], [], [], []
    for start in range(0, 14, 2):
        c = choice[:, start:start + 2]
        reflection = c % 2
        first = head(prediction['feature'], state[:, start:start + 2], reflection,
                     atlas, offsets, weights)
        selected = candidate_prediction(prediction, c, first['state'])
        index = torch.arange(2, device='cuda')[None]
        mapped64 = model.map(selected, offsets, index, reflection, (64, 64), atlas, weights,
                             return_refinement_feature=True, feature_side=64,
                             source_shape=(side, side), spatial_evidence=first['spatial_evidence'])
        second = head(prediction['feature'], first['state'], reflection, atlas, offsets,
                      weights, fit_summary=fit_summary(mapped64))
        selected['state'] = second['state']
        mapped96 = model.map(selected, offsets, index, reflection, (96, 96), atlas, weights,
                             feature_side=96, source_shape=(side, side),
                             spatial_evidence=second['spatial_evidence'])
        score = model.score_fitted_candidates(image, selected, mapped96, atlas,
                                              weights) + second['quality_logit']
        states.append(second['state'])
        scores.append(score)
        surfaces.append(mapped96['centre_surface_ccf_ap_dv_ml_um'])
        evidence.append(second['spatial_evidence'])
    return (prediction, choice, torch.cat(states, 1), torch.cat(scores, 1),
            torch.cat(surfaces, 1), torch.cat(evidence, 1))


def field_at_chart(surface, chart):
    count = surface.shape[1]
    grid = (chart + .5 / side) * 2 - 1
    grid = grid[None].expand(count, -1, -1).reshape(count, 1, len(chart), 2)
    field = surface[0].permute(0, 3, 1, 2)
    return F.grid_sample(field, grid, padding_mode='border',
                         align_corners=False).squeeze(2).transpose(1, 2)


def measure(image, offsets, weights, model, head, atlas, chart, reference,
            full_target=None, valid=None):
    prediction, choice, state, score, surface, evidence = infer(
        image, offsets, weights, model, head, atlas)
    reflection = choice % 2
    rigid = (points(state, reflection, chart) - reference).norm(dim=-1).mean(-1)[0]
    mapped96 = (field_at_chart(surface, chart) - reference[None]).norm(dim=-1).mean(-1)
    selected_index = int(score[0].argmax())
    selected_choice = choice[:, selected_index:selected_index + 1]
    selected = candidate_prediction(prediction, selected_choice,
        state[:, selected_index:selected_index + 1])
    mapped256 = model.map(selected, offsets,
        torch.zeros((1, 1), device='cuda', dtype=torch.long),
        reflection[:, selected_index:selected_index + 1], (side, side), atlas, weights,
        source_shape=(side, side),
        spatial_evidence=evidence[:, selected_index:selected_index + 1])
    field = mapped256['centre_surface_ccf_ap_dv_ml_um'][0, 0]
    if valid is None:
        final_error = (field_at_chart(field[None, None], chart)[0]
                       - reference).norm(dim=-1).mean()
    else:
        final_error = (field[valid] - full_target).norm(dim=-1).mean()
    return {'selected_mapped256_um': float(final_error),
            'selected_mapped96_um': float(mapped96[selected_index]),
            'best14_mapped96_um': float(mapped96.min()),
            'selected_rigid_um': float(rigid[selected_index]),
            'best14_rigid_um': float(rigid.min()),
            'selected_branch': int(choice[0, selected_index]),
            'selected_from_anchor': bool(int(choice[0, selected_index]) >= 32)}


def identity_mean(group, field, identity):
    return float(np.mean([np.mean([row[field] for row in group if row[identity] == key])
                          for key in {row[identity] for row in group}]))


training = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
steps = tuple(training['checkpoints'])
assert finished['batches'] == training['batches'] == 1000 and steps == (0, 500, 1000)
assert not finished['calibrated'] and not finished['public_benchmark_used']
assert sha(run / 'config.json') == finished['config_sha256']
assert sha(run / 'draws.jsonl') == finished['draws_sha256']
assert sha(run / 'training.jsonl') == finished['training_sha256']
assert all(sha(run / f'ranker_step_{int(step):05d}.pt') == digest
           for step, digest in finished['checkpoint_sha256'].items())
assert sha(training['parent']) == training['parent_sha256']
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in training['source_sha256'].items())
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in training['synthetic_provenance']['source_sha256'].items())
assert sha(root / 'runs/dense_atlas_correspondence_083_pilot/completed.json') \
       == training['pilot_083_completed_sha256']
assert sha(root / 'runs/dense_atlas_correspondence_083_pilot/match_step_06000.pt') \
       == training['pilot_083_checkpoint_6000_sha256']
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
synthetic = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic) == 246 and len({row['synthetic_subject_plan_id'] for row in synthetic}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
prior_receipt = json.loads((prior_eval / 'completed.json').read_text())
prior_config = json.loads((prior_eval / 'config.json').read_text())
assert sha(prior_eval / 'config.json') == prior_receipt['config_sha256']
assert sha(prior_eval / 'rows.jsonl') == prior_receipt['rows_sha256']
assert sha(prior_eval / 'summary.json') == prior_receipt['summary_sha256']
assert sha(panel / 'records.jsonl') == prior_config['synthetic_records_sha256']
assert sha(real / 'records.jsonl') == prior_config['real_records_sha256']
assert sha(real / 'images.npy') == prior_config['real_images_sha256']
assert sha(real / 'geometry.npz') == prior_config['real_geometry_sha256']
baseline = {(row['set'], row['section_id']): row for row in
            map(json.loads, (prior_eval / 'rows.jsonl').open()) if row['step'] == 0}
assert len(baseline) == len(synthetic) + len(real_records)

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'training_completed_sha256': sha(run / 'completed.json'),
    'training_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': finished['checkpoint_sha256'],
    'evaluator_sha256': sha(__file__),
    'synthetic_panel_receipt_sha256': sha(panel / 'completed.json'),
    'synthetic_records_sha256': sha(panel / 'records.jsonl'),
    'real_records_sha256': sha(real / 'records.jsonl'),
    'real_images_sha256': prior_config['real_images_sha256'],
    'real_geometry_sha256': prior_config['real_geometry_sha256'],
    'parent_baseline_receipt_sha256': sha(prior_eval / 'completed.json'),
    'parent_baseline_rows_sha256': prior_receipt['rows_sha256'],
    'steps': steps, 'candidate_beam': {'old': 8, 'anchor': 6},
    'synthetic_selected_metric': 'mapped256 mean 3-D CCF error across all surviving tissue pixels',
    'synthetic_best14_metric': 'mapped96 truth-best among 14 on 1024 fixed surviving pixels; oracle diagnostic only',
    'real_reference': 'weak inherited Allen five-point affine, not verified anatomical truth',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
head = WholeSliceAtlasFeedback083().cuda().eval()
model.requires_grad_(False)
head.requires_grad_(False)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
support_cutoff = float(np.quantile([row['valid_pixels'] / side**2 for row in synthetic], .25))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(run / f'ranker_step_{step:05d}.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        head.load_state_dict(checkpoint['feedback'], strict=True)
        del checkpoint
        for split, records in (('synthetic', synthetic), ('real_weak_allen', real_records)):
            for record in records:
                previous = baseline[(split, record['section_id'])]
                if split == 'synthetic':
                    assert sha(panel / record['file']) == record['sha256']
                    assert previous['sha256'] == record['sha256']
                    with np.load(panel / record['file'], allow_pickle=False) as arrays:
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
                    full_target = target[valid]
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
                    valid, full_target = None, None
                current = measure(image, offsets, weights, model, head, atlas, chart,
                                  reference, full_target, valid)
                row = {'set': split, 'step': step, 'section_id': record['section_id'],
                       **{key: value for key, value in previous.items()
                          if key.startswith('parent_')},
                       **{'model_' + key: value for key, value in current.items()}}
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
            print(json.dumps({'step': step, 'set': split, 'sections': len(records)}), flush=True)

metrics = ('parent_selected_mapped256_um', 'model_selected_mapped256_um',
           'parent_selected_mapped96_um', 'model_selected_mapped96_um',
           'parent_best14_mapped96_um', 'model_best14_mapped96_um',
           'parent_selected_rigid_um', 'model_selected_rigid_um',
           'parent_best14_rigid_um', 'model_best14_rigid_um')
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
            for grouping, values in (('by_appearance', sorted({row['appearance_mode'] for row in group})),
                                     ('by_support', (False, True))):
                entry[split][grouping] = {}
                for value in values:
                    subset = [row for row in group if row['appearance_mode' if grouping ==
                        'by_appearance' else 'low_support'] == value]
                    entry[split][grouping][str(value)] = {
                        'sections': len(subset),
                        'identity_equal_mean_um': {metric: identity_mean(subset, metric, identity)
                                                   for metric in metrics}}
    if step == steps[-1]:
        values = entry['synthetic']['identity_equal_mean_um']
        gate = {'selected_gain_ge_250_um': values['parent_selected_mapped256_um']
                - values['model_selected_mapped256_um'] >= 250,
                'best14_le_110_percent_parent': values['model_best14_mapped96_um']
                <= 1.1 * values['parent_best14_mapped96_um']}
        gate['pass'] = all(gate.values())
        entry['gate'] = gate
    summary.append(entry)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
