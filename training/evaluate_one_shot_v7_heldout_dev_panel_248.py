"""Exploratory 192-pixel v7 DEV readout of the already selected 247 checkpoint."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_003'
TRAIN = ROOT / 'runs/anchor_only_blind_feedback_247_train_001'
READOUT_247 = ROOT / 'runs/anchor_only_blind_feedback_247_readout_002'
OUT = ROOT / 'runs/one_shot_v7_heldout_dev_panel_248_readout_001'
SIDE, FIRST_SIDE, FIT_SIDE, STEP = 192, 64, 96, 6000
DOC = Path('docs/publication/ONE_SHOT_V7_HELDOUT_DEV_PANEL_248_READOUT_PROTOCOL_20261012.md')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                (SIDE - 1) / SIDE - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi',
                                               frame[..., :2] @ basis, chart - .5)


assert (PANEL / 'completed.json').is_file(), '248 panel is not frozen; do not read partial output'
panel_completed = json.loads((PANEL / 'completed.json').read_text(encoding='utf8'))
assert sha(PANEL / 'protocol.json') == panel_completed['protocol_sha256']
assert sha(PANEL / 'records.jsonl') == panel_completed['records_sha256']
panel_protocol = json.loads((PANEL / 'protocol.json').read_text(encoding='utf8'))
assert panel_protocol['side'] == SIDE and panel_protocol['split'] == 'development'
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
assert len(records) == panel_completed['physical_sections'] == 256
assert len({row['provenance']['physical_section_id'] for row in records}) == 256
assert len({row['animal_id'] for row in records}) == panel_completed['synthetic_subjects'] == 8
assert all(row['split'] == 'development' and row['provenance']['split'] == 'development'
           for row in records)
panel_files = {row['file']: sha(PANEL / row['file']) for row in records}
assert all(panel_files[row['file']] == row['sha256'] for row in records)
subject_files = {row['file']: sha(PANEL / row['file']) for row in panel_completed['subjects']}
assert all(subject_files[row['file']] == row['sha256'] for row in panel_completed['subjects'])
eligible = [row for row in records if row['eligible'] and row['valid_pixels'] > 0]
assert len(eligible) == panel_completed['eligible']
training_completed = json.loads((TRAIN / 'completed.json').read_text(encoding='utf8'))
assert sha(TRAIN / 'config.json') == training_completed['config_sha256']
assert sha(TRAIN / f'joint_step_{STEP:05d}.pt') == training_completed['checkpoint_sha256'][str(STEP)]
readout_247 = json.loads((READOUT_247 / 'completed.json').read_text(encoding='utf8'))
assert sha(READOUT_247 / 'chosen_checkpoint.json') == readout_247['chosen_checkpoint_sha256']
chosen_247 = json.loads((READOUT_247 / 'chosen_checkpoint.json').read_text(encoding='utf8'))
assert readout_247['chosen_step'] == chosen_247['step'] == STEP
assert chosen_247['checkpoint_sha256'] == training_completed['checkpoint_sha256'][str(STEP)]
training_config = json.loads((TRAIN / 'config.json').read_text(encoding='utf8'))
assert training_config['source_sha256']['arbitrary_plane_one_shot_model.py'] == sha(
    Path(__file__).parent / 'arbitrary_plane_one_shot_model.py')
assert training_config['protocol_sha256'] == sha(
    Path('docs/publication/ANCHOR_ONLY_BLIND_FEEDBACK_247_PROTOCOL_20261012.md'))
assert torch.cuda.is_available()

atlas = torch.from_numpy(allen._decode_and_preprocess_allen_v6()[0]).cuda()
atlas_null = torch.stack((torch.roll(atlas[0], 80, 0), atlas[1]))
model = OneShotJointSliceModel(modes=0, normal_anchor_count=64,
                              shared_anchor_pose=True, atlas_conditioning=True,
                              vector_refinement=True, fitted_ranking=True).cuda().eval()
checkpoint = torch.load(TRAIN / f'joint_step_{STEP:05d}.pt', map_location='cpu', weights_only=True)
assert checkpoint['step'] == STEP and not checkpoint['calibrated']
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
full_chart = torch.stack(torch.meshgrid(torch.arange(SIDE, device='cuda'),
                                       torch.arange(SIDE, device='cuda'), indexing='ij'), -1)[..., [1, 0]].float() / SIDE


def fit(image, offsets, weights, prediction, mode_indices, reflections, atlas_image):
    selected = {**prediction, 'state': prediction['state'][:, mode_indices],
                'log_mass': prediction['log_mass'][:, mode_indices],
                'reflection_logit': prediction['reflection_logit'][:, mode_indices]}
    branch = torch.arange(8, device='cuda')[None]
    first = model.map(selected, offsets, branch, reflections, (FIRST_SIDE, FIRST_SIDE),
                      atlas_image, weights, return_refinement_feature=True,
                      feature_side=FIRST_SIDE, source_shape=image.shape[-2:])
    refined_state, _, _ = model.refine(first['refinement_feature'], selected['state'])
    refined = {**selected, 'state': refined_state}
    mapped = model.map(refined, offsets, branch, reflections, (FIT_SIDE, FIT_SIDE),
                       atlas_image, weights, feature_side=FIT_SIDE,
                       source_shape=image.shape[-2:])
    fitted = model.score_fitted_candidates(image, refined, mapped, atlas_image, weights)[0]
    return refined_state, fitted


def evaluate(record):
    with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
        image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
        valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        visible = torch.from_numpy(arrays['visible_support'].copy()).cuda()
        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        true_reflection = int(arrays['reflection'])
        offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
        weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        assert bool(arrays['eligible']) == record['eligible']
    assert int(valid.sum()) == record['valid_pixels']
    prediction = model.predict(image)
    modes = prediction['log_mass'][0].topk(4).indices
    mode_indices = modes.repeat_interleave(2)
    reflections = torch.tensor([[0, 1] * 4], device='cuda')
    prior = (prediction['log_mass'][0, mode_indices]
             + torch.where(reflections[0].bool(),
                           F.logsigmoid(prediction['reflection_logit'][0, mode_indices]),
                           F.logsigmoid(-prediction['reflection_logit'][0, mode_indices])))
    refined, fitted = fit(image, offsets, weights, prediction,
                          mode_indices, reflections, atlas)
    null_refined, null_fitted = fit(image, offsets, weights, prediction,
                                   mode_indices, reflections, atlas_null)
    selected, prior_selected, null_selected = int(fitted.argmax()), int(prior.argmax()), int(null_fitted.argmax())
    indices = valid.flatten().nonzero().flatten()
    chart = full_chart.reshape(-1, 2)[indices]
    target = reference.reshape(-1, 3)[indices]
    initial_rigid = (points(prediction['state'][:, mode_indices], reflections, chart)
                     - target).norm(dim=-1).mean(-1)[0]
    refined_rigid = (points(refined, reflections, chart) - target).norm(dim=-1).mean(-1)[0]
    null_rigid = (points(null_refined, reflections, chart) - target).norm(dim=-1).mean(-1)[0]
    selected_states = torch.cat((refined[:, selected:selected + 1], truth[:, None]), 1)
    selected_reflections = torch.tensor([[int(reflections[0, selected]), true_reflection]], device='cuda')
    mapped192 = model.map({**prediction, 'state': selected_states}, offsets,
                          torch.tensor([[0, 1]], device='cuda'), selected_reflections,
                          (SIDE, SIDE), atlas, weights, source_shape=image.shape[-2:])
    mapped192_error = (mapped192['centre_surface_ccf_ap_dv_ml_um'][0, :, valid]
                       - target[None]).norm(dim=-1).mean(-1)
    zero_warp = (points(truth[:, None], selected_reflections[:, 1:], chart)
                 - target).norm(dim=-1).mean()
    true_axis = full_frame_state_to_components(truth)[1][0, :, 2]
    cut_axis = torch.as_tensor(record['provenance']['virtual_unit_normal'], device='cuda')
    initial_axes = full_frame_state_to_components(prediction['state'][:, mode_indices])[1][0, :, :, 2]
    refined_axes = full_frame_state_to_components(refined)[1][0, :, :, 2]
    initial_axis_error = torch.rad2deg(torch.acos(
        (initial_axes[prior_selected] * true_axis).sum().abs().clamp_max(1)))
    selected_axis_error = torch.rad2deg(torch.acos(
        (refined_axes[selected] * true_axis).sum().abs().clamp_max(1)))
    cut_nearest_principal = torch.rad2deg(torch.acos(cut_axis.abs().max().clamp_max(1)))
    return {'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
            'physical_section_id': record['provenance']['physical_section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'synthetic_subject_realization_id': record['synthetic_subject_realization_id'],
            'plan_receipt_sha256': record['plan_receipt_sha256'],
            'inverse_map_sha256': record['inverse_map_sha256'],
            'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
            'artifacts': record['provenance']['one_shot_slide_artifacts_v3']['events'],
            'cut_nearest_principal_axis_deg': float(cut_nearest_principal),
            'source_visible_support_fraction': record['source_visible_support_mass'] / SIDE**2,
            'observed_visible_support_fraction': float(visible.sum()) / SIDE**2,
            'valid_fraction': record['valid_pixels'] / SIDE**2,
            'prior_top1_normal_axis_error_deg': float(initial_axis_error),
            'fitted_selected_normal_axis_error_deg': float(selected_axis_error),
            'prior_top1_rigid_um': float(initial_rigid[prior_selected]),
            'prior_top1_refined_rigid_um': float(refined_rigid[prior_selected]),
            'prior_top8_best_rigid_um': float(initial_rigid.min()),
            'refined_top8_best_rigid_um': float(refined_rigid.min()),
            'fitted_selected_initial_rigid_um': float(initial_rigid[selected]),
            'fitted_selected_rigid_um': float(refined_rigid[selected]),
            'null_selected_rigid_um': float(null_rigid[null_selected]),
            'fitted_selected_mapped_full192_um': float(mapped192_error[0]),
            'exact_pose_mapped_full192_um': float(mapped192_error[1]),
            'exact_pose_zero_warp_full192_um': float(zero_warp),
            'prior_top1_mode': int(mode_indices[prior_selected]),
            'fitted_selected_mode': int(mode_indices[selected]),
            'null_selected_mode': int(mode_indices[null_selected]),
            'prior_top1_reflection': int(reflections[0, prior_selected]),
            'fitted_selected_reflection': int(reflections[0, selected])}


OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        row = evaluate(record)
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'evaluated': len(rows), 'eligible': len(eligible)}), flush=True)

metrics = ('prior_top1_rigid_um', 'prior_top1_refined_rigid_um',
           'prior_top8_best_rigid_um', 'refined_top8_best_rigid_um',
           'fitted_selected_initial_rigid_um', 'fitted_selected_rigid_um',
           'null_selected_rigid_um',
           'fitted_selected_mapped_full192_um', 'exact_pose_mapped_full192_um',
           'exact_pose_zero_warp_full192_um', 'prior_top1_normal_axis_error_deg',
           'fitted_selected_normal_axis_error_deg')


def summarize(group):
    identities = sorted({row['animal_id'] for row in group})
    subject_means = {identity: {key: float(np.mean([row[key] for row in group
                                                  if row['animal_id'] == identity])) for key in metrics}
                     for identity in identities}
    return {'cases': len(group), 'synthetic_subjects': len(identities),
            'cases_per_subject': {identity: sum(row['animal_id'] == identity for row in group)
                                  for identity in identities},
            'case_median': {key: float(np.median([row[key] for row in group])) for key in metrics} if group else {},
            'subject_equal_mean': {key: float(np.mean([subject_means[identity][key] for identity in identities]))
                                   for key in metrics} if group else {},
            'subject_means': subject_means}


summary = {
    'status': 'exploratory DEV diagnostic; no promotion gate or checkpoint selection on 248 panel',
    'attempted': len(records), 'eligible': len(rows), 'ineligible': len(records) - len(rows),
    'checkpoint_step': STEP, 'all_eligible': summarize(rows),
    'physical_cut_nearest_principal_axis_deg': {
        f'{low}-{high}': summarize([row for row in rows
                                   if low <= row['cut_nearest_principal_axis_deg'] < high])
        for low, high in ((0, 15), (15, 30), (30, 45), (45, 60))},
    'observed_visible_support_fraction': {
        f'{low:.1f}-{high:.1f}': summarize([row for row in rows
                                           if low <= row['observed_visible_support_fraction'] < high])
        for low, high in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01))},
    'metric_note': 'All rigid and map errors use every observed valid 192-pixel site. The shifted null '
                   'rolls atlas intensity 80 voxels along AP, preserves support and the source prior beam, '
                   'and reruns mapping, refinement and scoring. Exact-pose map is fed the target affine '
                   'plane but still uses predicted image features; zero-warp is the same target plane '
                   'without the learned local field. Eight synthetic subjects share one Allen atlas.'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'status': summary['status'], 'rows': len(rows), 'checkpoint_step': STEP,
    'checkpoint_sha256': sha(TRAIN / f'joint_step_{STEP:05d}.pt'),
    'training_config_sha256': sha(TRAIN / 'config.json'),
    'training_completed_sha256': sha(TRAIN / 'completed.json'),
    'selected_247_readout_completed_sha256': sha(READOUT_247 / 'completed.json'),
    'selected_247_chosen_checkpoint_sha256': sha(READOUT_247 / 'chosen_checkpoint.json'),
    'panel_sha256': {name: sha(PANEL / name) for name in ('protocol.json', 'records.jsonl', 'completed.json')},
    'panel_file_sha256': panel_files, 'panel_subject_file_sha256': subject_files,
    'readout_protocol_sha256': sha(DOC),
    'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'atlas_normalized_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in (
        Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_allen_atlas_binding_v6.py')},
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'evaluated': len(rows), 'attempted': len(records)}), flush=True)
