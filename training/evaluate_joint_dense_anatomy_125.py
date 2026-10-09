"""Fresh-section 125 development readout; frozen 122 and final 125 only."""

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
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.atlas_anatomy_ranker_125 import (
    atlas_anatomy_rank_125, make_atlas_anatomy_ranker_125)
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/JOINT_DENSE_ANATOMY_125_PROTOCOL_20261009.md'
parent_run = root / 'runs/joint_pose_map_122'
parent_path = parent_run / 'joint_step_02000.pt'
prior_ranker_run = root / 'runs/joint_anatomy_ranker_124'
run = root / 'runs/joint_dense_anatomy_125'
ranker_path = run / 'ranker_step_10000.pt'
panel = root / 'data/joint_dense_anatomy_125_dev_panel'
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
reserved = root / 'data/joint_v7_reserved_train_images_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
out = root / 'runs/joint_dense_anatomy_125_dev_eval'
side, map_side = 256, 96
arms = ('frozen122', 'full', 'support', 'zero_intensity', 'intensity_shuffle', 'source_swap')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
train_done = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
parent_done = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert train_done['presentations'] == train_config['presentations'] == 10000
assert tuple(train_config['checkpoints']) == (0, 2500, 5000, 7500, 10000)
assert train_config['candidate_count'] == 16 and train_config['fixed_psf_um'] == 62.5
assert train_config['ranker_side'] == 32 and train_config['psf_samples'] == 9
assert sha(protocol) == train_done['protocol_sha256'] == train_config['protocol_sha256']
assert sha(run / 'config.json') == train_done['config_sha256']
assert sha(run / 'draws.jsonl') == train_done['draws_sha256']
assert sha(run / 'training.jsonl') == train_done['training_sha256']
assert all(sha(run / f'ranker_step_{step:05d}.pt') ==
           train_done['checkpoint_sha256'][str(step)] for step in
           train_config['checkpoints'])
assert train_done['source_sha256'] == train_config['source_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000'] == \
    train_done['parent_checkpoint_sha256'] == train_config['parent_checkpoint_sha256']
assert sha(parent_run / 'completed.json') == train_config['parent_completion_sha256']
assert sha(parent_run / 'config.json') == parent_done['config_sha256'] == \
    train_config['parent_config_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_done['draws_sha256'] == \
    train_config['parent_draws_sha256']
assert sha(prior_ranker_run / 'completed.json') == train_config['prior_124_completion_sha256']
assert sha(prior_ranker_run / 'draws.jsonl') == \
    train_config['prior_124_draws_sha256'] == train_done['prior_124_draws_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(train_done.get(key, False) or train_config.get(key, False)
               or parent_done.get(key, False) for key in (
                   'calibrated', 'expert_real_truth_used', 'final_animals_used',
                   'public_benchmark_used', 'external_pretrained_weights_used'))

panel_done = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert panel_protocol['seed'] == 2026100912501
assert panel_protocol['augmentation_seed_prefix'] == 2026100912502
assert panel_protocol['dense_125_protocol_sha256'] == sha(protocol)
assert sha(root / 'data/v3_pose_capture_confirmation_plans_001/completed.json') == \
    panel_protocol['source_plan_completed_sha256'] == panel_done['plan_completed_sha256']
assert all(sha(root / 'data' / name / 'records.jsonl') == digest
           for name, digest in panel_protocol['prior_panel_records_sha256'].items())
assert all(sha(panel / 'source' / name) == digest
           for name, digest in panel_protocol['source_sha256'].items())
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = sorted((row for row in records if row['eligible']),
                  key=lambda row: (row['synthetic_subject_plan_id'], row['section_id']))
plans = sorted({row['synthetic_subject_plan_id'] for row in eligible})
assert len(records) == panel_done['physical_sections'] == 256
assert sum(not row['eligible'] for row in records) == panel_done['ineligible']
assert len(eligible) == panel_done['eligible'] and len(plans) == 8
assert len({row['section_id'] for row in records}) == len(records)
assert len({row['panel_physical_section_id'] for row in records}) == len(records)
assert all(row['section_id'].split('-joint-dense-125-section-')[-1].isdigit()
           and 'joint-dense-125-section-' in row['section_id'] for row in records)
assert all(row['provenance']['split'] == 'development' and
           sha(panel / row['file']) == row['sha256'] for row in records)
assert all(sum(row['synthetic_subject_plan_id'] == plan for row in records) == 32
           for plan in plans)
old_panel = root / 'data/v3_pose_capture_confirmation_panel_001'
old_sections = [json.loads(line) for line in (old_panel / 'records.jsonl').open()]
assert not {row['section_id'] for row in records} & {
    row['section_id'] for row in old_sections}
assert not {row['panel_physical_section_id'] for row in records} & {
    row['panel_physical_section_id'] for row in old_sections}
plan_rows = {plan: [row for row in eligible if row['synthetic_subject_plan_id'] == plan]
             for plan in plans}
swap_record = {}
for index, plan in enumerate(plans):
    next_rows = plan_rows[plans[(index + 1) % len(plans)]]
    for position, record in enumerate(plan_rows[plan]):
        swap_record[record['section_id']] = next_rows[position % len(next_rows)]

train_draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
assert len(train_draws) == train_done['synthetic_draw_attempts']
assert sum(row['used'] for row in train_draws) == train_done['presentations']
assert all(row['base_lineage']['split'] == 'train' for row in train_draws)
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in records} & {
        row['base_lineage'][key] for row in train_draws}
assert not {row['panel_physical_section_id'] for row in records} & {
    row['physical_section_id'] for row in train_draws}
parent_draws = [row for row in map(json.loads, (parent_run / 'draws.jsonl').open())
                if 'physical_section_id' in row]
assert not {row['panel_physical_section_id'] for row in records} & {
    row['physical_section_id'] for row in parent_draws}
prior_draws = [row for row in map(json.loads,
               (prior_ranker_run / 'draws.jsonl').open()) if 'physical_section_id' in row]
assert not {row['panel_physical_section_id'] for row in records} & {
    row['physical_section_id'] for row in prior_draws}

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert coronal_done['development_images'] == 64 and coronal_done['development_donors'] == 6
assert all(sha(coronal / name) == coronal_done['output_sha256'][name]
           for name in ('images.npy', 'geometry.npz', 'records.jsonl'))
coronal_all = [json.loads(line) for line in (coronal / 'records.jsonl').open()]
coronal_records = [row for row in coronal_all if row['training_split'] == 'development']
coronal_train = [row for row in coronal_all if row['training_split'] == 'train']
assert len(coronal_records) == 64 and len({row['animal_id'] for row in coronal_records}) == 6
assert not {row['animal_id'] for row in coronal_records} & {
    row['animal_id'] for row in coronal_train}
reserved_done = json.loads((reserved / 'completed.json').read_text())
assert reserved_done['training_role'] == 'train_only'
assert not {str(row['animal_id']) for row in coronal_records} & set(reserved_done['donor_receipts'])
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()

sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert sagittal_summary['roles']['weak_dev2_source_available'] == {
    'donors': 8, 'sections': 158}
assert all(sha(sagittal / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_train_summary = json.loads((sagittal_train / 'summary.json').read_text())
assert sha(sagittal_train / 'summary.json') == parent_config['sagittal_summary_sha256']
assert sha(sagittal_train / 'geometry.jsonl') == \
    sagittal_train_summary['output_sha256']['geometry.jsonl']
sagittal_train_records = [json.loads(line) for line in
                          (sagittal_train / 'geometry.jsonl').open()]
assert len(sagittal_records) == 158 and len({row['donor_id'] for row in sagittal_records}) == 8
assert all(row['split'] == 'weak_dev2_source_available' for row in sagittal_records)
assert all(row['split'] == 'train' for row in sagittal_train_records)
assert not {row['donor_id'] for row in sagittal_records} & {
    row['donor_id'] for row in sagittal_train_records}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance'] == \
    train_config['synthetic_provenance']
atlas = context['atlas']
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['config'] == parent_config and not parent['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(parent['model'], strict=True)
del parent
checkpoint = torch.load(ranker_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 10000 and checkpoint['config'] == train_config
assert checkpoint['calibrated'] is False
full_ranker = make_atlas_anatomy_ranker_125().cuda().eval().requires_grad_(False)
support_ranker = make_atlas_anatomy_ranker_125().cuda().eval().requires_grad_(False)
full_ranker.load_state_dict(checkpoint['full_ranker'], strict=True)
support_ranker.load_state_dict(checkpoint['support_ranker'], strict=True)
del checkpoint

psf_offsets = torch.linspace(-31.25, 31.25, 9, device='cuda')[None]
psf_weights = torch.tensor([[1., 2., 2., 2., 2., 2., 2., 2., 1.]], device='cuda') / 16
axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
five_pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                            [127.5, 127.5]], device='cuda')
five_chart = five_pixels / side


def forward(image):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
        similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity = -similarity
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    mode, reflection = beam // 2, beam % 2
    match = global_plane_match(model, prediction, mode, reflection,
        psf_offsets, psf_weights, atlas, (side, side), side=24)
    mapped = model.map({**prediction, 'state': match['state']}, psf_offsets,
        torch.arange(16, device='cuda')[None], reflection, (map_side, map_side),
        atlas, psf_weights, source_shape=(side, side))
    return prediction, beam, reflection, match, mapped


synthetic_rows, real_rows = [], []
with torch.inference_mode():
    for record in eligible:
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            truth_reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
        with np.load(panel / swap_record[record['section_id']]['file'], allow_pickle=False) as arrays:
            swapped_image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        prediction, beam, reflection, match, mapped = forward(image)
        source_prediction = {'feature': model.predict(swapped_image)['feature']}
        shuffle_seed = int.from_bytes(hashlib.sha256(record['section_id'].encode()).digest()[:8],
                                      'little') % (2 ** 63)
        fitted = {
            'full': atlas_anatomy_rank_125(full_ranker, prediction, image,
                match['state'], reflection, atlas,
                mapped_coordinates=mapped['coordinates'],
                local_displacement_um=mapped['local_displacement_um']),
            'support': atlas_anatomy_rank_125(support_ranker, prediction, image,
                match['state'], reflection, atlas,
                mapped_coordinates=mapped['coordinates'],
                local_displacement_um=mapped['local_displacement_um'], support_only=True),
            'zero_intensity': atlas_anatomy_rank_125(full_ranker, prediction, image,
                match['state'], reflection, atlas,
                mapped_coordinates=mapped['coordinates'],
                local_displacement_um=mapped['local_displacement_um'], support_only=True),
            'intensity_shuffle': atlas_anatomy_rank_125(full_ranker, prediction, image,
                match['state'], reflection, atlas,
                mapped_coordinates=mapped['coordinates'],
                local_displacement_um=mapped['local_displacement_um'],
                shuffle_intensity=True, shuffle_seed=shuffle_seed),
            'source_swap': atlas_anatomy_rank_125(full_ranker, source_prediction,
                swapped_image, match['state'], reflection, atlas,
                mapped_coordinates=mapped['coordinates'],
                local_displacement_um=mapped['local_displacement_um'])}
        assert all(torch.equal(fitted['full']['pair_valid'], fitted[arm]['pair_valid'])
                   and torch.equal(fitted['full']['centre_support'],
                                   fitted[arm]['centre_support'])
                   for arm in ('support', 'zero_intensity', 'intensity_shuffle',
                               'source_swap'))
        scores = torch.stack((match['input_score'], match['score']), -1)
        choices = {'frozen122': divmod(int(scores.flatten().argmax()), 2)}
        choices.update({arm: divmod(int((scores + fitted[arm]['evidence'][..., None])
            .flatten().argmax()), 2) for arm in arms[1:]})
        target_rigid = rigid_points_090(truth, truth_reflection, chart)[0]
        original_rigid = (rigid_points_090(match['input_state'], reflection, chart)[0]
            - target_rigid).norm(dim=-1)[:, valid[0].flatten()].mean(-1) / 1000
        corrected_rigid = (rigid_points_090(match['state'], reflection, chart)[0]
            - target_rigid).norm(dim=-1)[:, valid[0].flatten()].mean(-1) / 1000
        surface = F.interpolate(mapped['centre_surface_ccf_ap_dv_ml_um'][0]
            .permute(0, 3, 1, 2), (side, side), mode='bilinear', align_corners=False)
        corrected_map = (surface.permute(0, 2, 3, 1) - centre[0]).norm(dim=-1)
        corrected_map = corrected_map[:, valid[0]].mean(-1) / 1000
        original_map = {}
        for slot in sorted({slot for slot, action in choices.values() if action == 0}):
            original = model.map({**prediction,
                'state': match['input_state'][:, slot:slot + 1]}, psf_offsets,
                torch.zeros((1, 1), device='cuda', dtype=torch.long),
                reflection[:, slot:slot + 1], (map_side, map_side), atlas,
                psf_weights, source_shape=(side, side))
            original_surface = F.interpolate(
                original['centre_surface_ccf_ap_dv_ml_um'][0, 0]
                .permute(2, 0, 1)[None], (side, side), mode='bilinear',
                align_corners=False)[0].permute(1, 2, 0)
            original_map[slot] = float((original_surface - centre[0])
                .norm(dim=-1)[valid[0]].mean() / 1000)
        selected = {arm: {'slot': slot,
            'action': 'original' if action == 0 else 'corrected',
            'rigid_gauge_mm': float((original_rigid if action == 0 else corrected_rigid)[slot]),
            'mapped_tissue_mm': original_map[slot] if action == 0
                else float(corrected_map[slot])}
            for arm, (slot, action) in choices.items()}

        best = int(corrected_map.argmin())
        pair = {'best_slot': best, 'best_mapped_mm': float(corrected_map[best]),
                'near_candidate': bool(float(corrected_map[best]) <= 1.5),
                'status': 'no_near_blind_candidate'}
        if pair['near_candidate']:
            support_mask = fitted['full']['centre_support'][0]
            overlap = (support_mask & support_mask[best]).sum((-2, -1)).float()
            dice = 2 * overlap / (support_mask.sum((-2, -1))
                + support_mask[best].sum()).clamp_min(1)
            area_gap = (support_mask.float().mean((-2, -1))
                - support_mask[best].float().mean()).abs()
            support_matched = ((corrected_map > corrected_map[best] + 1.)
                & (dice >= .85) & (area_gap <= .08))
            pair['support_matched_wrong_count'] = int(support_matched.sum())
            pair['status'] = 'no_support_matched_wrong'
            if bool(support_matched.any()):
                pair['max_common_pixels'] = int(
                    fitted['full']['pair_common_count'][0, best, support_matched].max())
                matched = support_matched & fitted['full']['pair_valid'][0, best]
                pair['common_interior_wrong_count'] = int(matched.sum())
                pair['status'] = 'insufficient_common_interior'
                if bool(matched.any()):
                    wrong = int(corrected_map.masked_fill(~matched, torch.inf).argmin())
                    pair.update(status='scored', wrong_slot=wrong,
                        wrong_mapped_mm=float(corrected_map[wrong]),
                        mapped_support_dice=float(dice[wrong]),
                        mapped_support_area_gap=float(area_gap[wrong]),
                        common_pixels=int(fitted['full']['pair_common_count'][0, best, wrong]),
                        margins={'frozen122': float(scores[0, best, 1] - scores[0, wrong, 1]),
                            **{arm: float(fitted[arm]['pair_margin'][0, best, wrong])
                               for arm in arms[1:]}})
        synthetic_rows.append({'section_id': record['section_id'],
            'physical_section_id': record['panel_physical_section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'appearance_mode': record['appearance_mode'],
            'panel_file_sha256': record['sha256'],
            'swap_section_id': swap_record[record['section_id']]['section_id'],
            'shuffle_seed': shuffle_seed,
            'beam_branch_ids': beam[0].tolist(), 'pair': pair, 'selected': selected})

    for family, records_family in (('coronal', coronal_records),
                                   ('sagittal', sagittal_records)):
        for record in records_family:
            index = record['array_row_index']
            if family == 'coronal':
                native = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
                image = F.interpolate(torch.from_numpy(native).cuda(), (side, side),
                    mode='bilinear', align_corners=False)
                affine = torch.as_tensor(coronal_affines[index], device='cuda',
                                         dtype=torch.float32)
                reference = affine[:, 2] + (192 / side) * (
                    five_pixels[:, :1] * affine[:, 0] + five_pixels[:, 1:] * affine[:, 1])
                donor = record['animal_id']
            else:
                native = np.concatenate((np.asarray(sagittal_images[index], dtype=np.float32),
                    np.zeros((4, side, side), dtype=np.float32)))[None]
                image = torch.from_numpy(native).cuda()
                affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                                         device='cuda', dtype=torch.float32)
                reference = affine[:, 2] + (five_pixels[:, :1] * affine[:, 0]
                                            + five_pixels[:, 1:] * affine[:, 1])
                donor = record['donor_id']
            prediction, beam, reflection, match, mapped = forward(image)
            full = atlas_anatomy_rank_125(full_ranker, prediction, image,
                match['state'], reflection, atlas,
                mapped_coordinates=mapped['coordinates'],
                local_displacement_um=mapped['local_displacement_um'])
            support = atlas_anatomy_rank_125(support_ranker, prediction, image,
                match['state'], reflection, atlas,
                mapped_coordinates=mapped['coordinates'],
                local_displacement_um=mapped['local_displacement_um'], support_only=True)
            scores = torch.stack((match['input_score'], match['score']), -1)
            choices = {'frozen122': divmod(int(scores.flatten().argmax()), 2),
                'full': divmod(int((scores + full['evidence'][..., None]).flatten().argmax()), 2),
                'support': divmod(int((scores + support['evidence'][..., None]).flatten().argmax()), 2)}
            selected = {}
            for arm, (slot, action) in choices.items():
                state = (match['input_state'] if action == 0 else match['state'])[:, slot]
                proposed = rigid_points_090(state, reflection[:, slot], five_chart)[0]
                selected[arm] = {'slot': slot,
                    'action': 'original' if action == 0 else 'corrected',
                    'weak_five_point_mm': float((proposed - reference).norm(dim=-1).mean() / 1000)}
            real_rows.append({'family': family, 'donor_id': donor,
                'section_id': record['section_id'], 'beam_branch_ids': beam[0].tolist(),
                'selected': selected,
                'full_changed_slot': selected['full']['slot'] != selected['frozen122']['slot'],
                'full_changed_action': selected['full']['action'] != selected['frozen122']['action'],
                'support_changed_slot': selected['support']['slot'] != selected['frozen122']['slot'],
                'support_changed_action': selected['support']['action'] != selected['frozen122']['action'],
                'label_role': 'inherited weak Allen affine only; not expert truth'})

assert len(synthetic_rows) == len(eligible)
assert len(real_rows) == len(coronal_records) + len(sagittal_records) == 222
scored = [row for row in synthetic_rows if row['pair']['status'] == 'scored']
pair_wins = {arm: float(np.mean([row['pair']['margins'][arm] >= .01 for row in scored]))
             if scored else None for arm in arms}
by_plan = {plan: [row for row in synthetic_rows
                  if row['synthetic_subject_plan_id'] == plan] for plan in plans}
synthetic_summary = {'sections': len(synthetic_rows), 'plans': len(plans),
    'panel_physical_sections': len(records),
    'panel_ineligible_sections': panel_done['ineligible'],
    'status_counts': {status: sum(row['pair']['status'] == status for row in synthetic_rows)
                      for status in sorted({row['pair']['status'] for row in synthetic_rows})},
    'near_candidate_count': sum(row['pair']['near_candidate'] for row in synthetic_rows),
    'scored_pairs': len(scored), 'pair_win_fraction': pair_wins,
    'pair_margin': {arm: {'mean': float(np.mean([row['pair']['margins'][arm]
        for row in scored])) if scored else None,
        'median': float(np.median([row['pair']['margins'][arm]
        for row in scored])) if scored else None} for arm in arms},
    'common_pixels_median': float(np.median([row['pair']['common_pixels']
        for row in scored])) if scored else None,
    'by_plan': {}, 'section_weighted': {}, 'plan_equal': {},
    'support_dice_area_common_strata': {}, 'selected_changes': {}}
for metric, limits in (('mapped_support_dice', (.85, .90, .95, 1.000001)),
                       ('mapped_support_area_gap', (0., .02, .05, .080001)),
                       ('common_pixels', (32, 64, 128, 1025))):
    synthetic_summary['support_dice_area_common_strata'][metric] = {}
    for low, high in zip(limits[:-1], limits[1:]):
        group = [row for row in scored if low <= row['pair'][metric] < high]
        synthetic_summary['support_dice_area_common_strata'][metric][f'[{low},{high})'] = {
            'pairs': len(group),
            'pair_win_fraction': {arm: float(np.mean([
                row['pair']['margins'][arm] >= .01 for row in group])) if group else None
                for arm in arms}}
for arm in arms[1:]:
    synthetic_summary['selected_changes'][arm] = {
        'slot': sum(row['selected'][arm]['slot'] != row['selected']['frozen122']['slot']
                    for row in synthetic_rows),
        'action': sum(row['selected'][arm]['action'] != row['selected']['frozen122']['action']
                      for row in synthetic_rows)}
for plan, group in by_plan.items():
    plan_scored = [row for row in group if row['pair']['status'] == 'scored']
    synthetic_summary['by_plan'][plan] = {'sections': len(group),
        'panel_physical_sections': 32, 'panel_ineligible_sections': 32 - len(group),
        'scored_pairs': len(plan_scored),
        'pair_win_fraction': {arm: float(np.mean([
            row['pair']['margins'][arm] >= .01 for row in plan_scored]))
            if plan_scored else None for arm in arms},
        'selected_changes': {arm: {
            'slot': sum(row['selected'][arm]['slot'] != row['selected']['frozen122']['slot']
                        for row in group),
            'action': sum(row['selected'][arm]['action'] != row['selected']['frozen122']['action']
                          for row in group)} for arm in arms[1:]},
        'selected': {arm: {metric: float(np.mean([row['selected'][arm][metric]
            for row in group])) for metric in ('rigid_gauge_mm', 'mapped_tissue_mm')}
            for arm in arms}}
for arm in arms:
    synthetic_summary['section_weighted'][arm] = {metric: float(np.mean([
        row['selected'][arm][metric] for row in synthetic_rows]))
        for metric in ('rigid_gauge_mm', 'mapped_tissue_mm')}
    synthetic_summary['plan_equal'][arm] = {metric: float(np.mean([
        synthetic_summary['by_plan'][plan]['selected'][arm][metric] for plan in plans]))
        for metric in ('rigid_gauge_mm', 'mapped_tissue_mm')}
synthetic_summary['plans_full_mapped_not_worse_than_frozen122'] = sum(
    synthetic_summary['by_plan'][plan]['selected']['full']['mapped_tissue_mm'] <=
    synthetic_summary['by_plan'][plan]['selected']['frozen122']['mapped_tissue_mm']
    for plan in plans)

real_summary = {}
for family in ('coronal', 'sagittal'):
    group = [row for row in real_rows if row['family'] == family]
    donors = sorted({row['donor_id'] for row in group})
    by_donor = {str(donor): {'sections': len(donor_rows),
        'selected_weak_five_point_mm': {arm: float(np.mean([
            row['selected'][arm]['weak_five_point_mm'] for row in donor_rows]))
            for arm in ('frozen122', 'full', 'support')},
        'full_changed_slot': sum(row['full_changed_slot'] for row in donor_rows),
        'full_changed_action': sum(row['full_changed_action'] for row in donor_rows),
        'support_changed_slot': sum(row['support_changed_slot'] for row in donor_rows),
        'support_changed_action': sum(row['support_changed_action'] for row in donor_rows)}
        for donor in donors if (donor_rows := [row for row in group if row['donor_id'] == donor])}
    real_summary[family] = {'sections': len(group), 'donors': len(donors),
        'by_donor': by_donor,
        'donor_equal_weak_five_point_mm': {arm: float(np.mean([
            value['selected_weak_five_point_mm'][arm] for value in by_donor.values()]))
            for arm in ('frozen122', 'full', 'support')},
        'full_changed_slot': sum(row['full_changed_slot'] for row in group),
        'full_changed_action': sum(row['full_changed_action'] for row in group),
        'support_changed_slot': sum(row['support_changed_slot'] for row in group),
        'support_changed_action': sum(row['support_changed_action'] for row in group),
        'label_role': 'inherited weak Allen affine only; not expert truth'}

mapped = synthetic_summary['plan_equal']
conditions = {
    'matched_pairs_ge_64': len(scored) >= 64,
    'full_pair_wins_ge_0p75': pair_wins['full'] is not None and pair_wins['full'] >= .75,
    'full_over_support_pair_wins_ge_0p10': pair_wins['full'] is not None and
        pair_wins['full'] - pair_wins['support'] >= .10,
    'selected_mapped_gain_ge_0p30_vs_frozen122':
        mapped['frozen122']['mapped_tissue_mm'] - mapped['full']['mapped_tissue_mm'] >= .30,
    'selected_mapped_gain_ge_0p15_vs_support':
        mapped['support']['mapped_tissue_mm'] - mapped['full']['mapped_tissue_mm'] >= .15,
    'selected_rigid_no_regression_vs_frozen122':
        mapped['full']['rigid_gauge_mm'] <= mapped['frozen122']['rigid_gauge_mm'],
    'plans_not_worse_ge_6_of_8':
        synthetic_summary['plans_full_mapped_not_worse_than_frozen122'] >= 6,
    'zero_intensity_pair_benefit_not_reproduced': pair_wins['full'] is not None and
        pair_wins['full'] - pair_wins['zero_intensity'] >= .10,
    'intensity_shuffle_pair_benefit_not_reproduced': pair_wins['full'] is not None and
        pair_wins['full'] - pair_wins['intensity_shuffle'] >= .10,
    'source_swap_pair_benefit_not_reproduced': pair_wins['full'] is not None and
        pair_wins['full'] - pair_wins['source_swap'] >= .10}
for family in ('coronal', 'sagittal'):
    real = real_summary[family]
    conditions[f'{family}_donor_equal_regression_le_0p20'] = (
        real['donor_equal_weak_five_point_mm']['full'] <=
        real['donor_equal_weak_five_point_mm']['frozen122'] + .20)
    conditions[f'{family}_every_donor_regression_le_0p50'] = all(
        donor['selected_weak_five_point_mm']['full'] <=
        donor['selected_weak_five_point_mm']['frozen122'] + .50
        for donor in real['by_donor'].values())
summary = {'scope': 'fresh sections on reused synthetic DEV plans, plus existing weak-real DEV; development evidence only',
    'fixed_psf_um': 62.5, 'psf_samples': 9,
    'label_sampling_note': 'TRAIN uses dense valid-interior correspondence labels and a sampled-site approximation for blind pair labels; DEV pair labels use all valid original pixels.',
    'blind_selection': 'frozen 122 blind16 beam and matcher; candidate evidence added equally '
        'to original/corrected action logits; unsupported pairs give zero evidence',
    'synthetic': synthetic_summary, 'real_weak_affine': real_summary,
    'stage1_anatomical_evidence_gate': {'conditions': conditions,
        'passed': all(conditions.values()), 'development_only': True},
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
output_config = {'ranker_final_step': 10000, 'parent_step': 2000,
    'ranker_completion_sha256': sha(run / 'completed.json'),
    'ranker_config_sha256': sha(run / 'config.json'),
    'ranker_draws_sha256': sha(run / 'draws.jsonl'),
    'ranker_training_sha256': sha(run / 'training.jsonl'),
    'ranker_checkpoint_sha256': sha(ranker_path),
    'ranker_checkpoint_receipt_sha256': train_done['checkpoint_sha256'],
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'parent_config_sha256': sha(parent_run / 'config.json'),
    'parent_checkpoint_sha256': sha(parent_path),
    'prior_124_completion_sha256': sha(prior_ranker_run / 'completed.json'),
    'prior_124_draws_sha256': sha(prior_ranker_run / 'draws.jsonl'),
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'coronal_completion_sha256': sha(coronal / 'completed.json'),
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'sagittal_train_summary_sha256': sha(sagittal_train / 'summary.json'),
    'sagittal_train_geometry_sha256': sha(sagittal_train / 'geometry.jsonl'),
    'protocol_sha256': sha(protocol), 'evaluator_source_sha256': sha(__file__),
    'ranker_module_sha256': sha(source / 'atlas_anatomy_ranker_125.py'),
    'fixed_psf_um': 62.5, 'panel_eligible_sections': len(eligible),
    'real_coronal_sections': len(coronal_records),
    'real_sagittal_sections': len(sagittal_records),
    'zero_intensity_scope': 'full-intensity ranker with zero atlas intensity, same support/maps',
    'intensity_shuffle_scope': 'full ranker with seeded within-support spatial permutation '
        'of atlas intensity independently per candidate/depth; fixed atlas support/maps',
    'source_swap_scope': 'synthetic DEV source channel-0 and predicted feature from next '
        'development plan, original candidate states/maps/support retained',
    'dev_label_pixels': 'all valid original 256x256 pixels',
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(output_config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for name, rows in (('synthetic_rows.jsonl', synthetic_rows), ('real_weak_rows.jsonl', real_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            print(json.dumps(row, allow_nan=False), file=stream)
(out / 'completed.json').write_text(json.dumps({'eligible_synthetic_sections': len(eligible),
    'real_coronal_sections': len(coronal_records),
    'real_sagittal_sections': len(sagittal_records),
    'scored_pairs': len(scored),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'summary.json', 'synthetic_rows.jsonl', 'real_weak_rows.jsonl')},
    'ranker_checkpoint_sha256': output_config['ranker_checkpoint_sha256'],
    'parent_checkpoint_sha256': output_config['parent_checkpoint_sha256'],
    'protocol_sha256': output_config['protocol_sha256'],
    'evaluator_source_sha256': output_config['evaluator_source_sha256'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'completed': True,
    'scored_pairs': len(scored),
    'stage1_anatomical_evidence_gate': summary['stage1_anatomical_evidence_gate']},
    allow_nan=False), flush=True)
