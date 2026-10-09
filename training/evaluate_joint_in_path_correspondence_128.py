"""Frozen 122 versus matched 128 control/treatment on the fresh synthetic DEV panel."""

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
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match
from training.in_path_global_matcher_128 import global_plane_match_128


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/JOINT_IN_PATH_CORRESPONDENCE_128_PROTOCOL_20261010.md'
design = source.parent / 'docs/publication/JOINT_IN_PATH_CORRESPONDENCE_128_DESIGN_20261010.md'
parent_run = root / 'runs/joint_pose_map_122'
parent_path = parent_run / 'joint_step_02000.pt'
manifest = root / 'runs/joint_in_path_correspondence_128_manifest'
run_dirs = {'control': root / 'runs/joint_in_path_correspondence_128_control',
            'treatment': root / 'runs/joint_in_path_correspondence_128_treatment'}
panel = root / 'data/joint_in_path_correspondence_128_dev_panel'
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
reserved = root / 'data/joint_v7_reserved_train_images_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
out = root / 'runs/joint_in_path_correspondence_128_dev_eval'
side, map_side = 256, 96
arm_names = ('frozen122', 'control', 'treatment')
ablation_names = ('geometry_only', 'atlas_intensity_zero',
                  'atlas_intensity_shuffle', 'source_features_zero', 'source_swap')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
parent_done = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_run / 'config.json') == parent_done['config_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_done['draws_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
manifest_done = json.loads((manifest / 'completed.json').read_text())
assert sha(manifest / 'draws.jsonl') == manifest_done['draws_sha256']
assert sha(manifest / 'updates.jsonl') == manifest_done['updates_sha256']
assert sha(manifest / 'config.json') == manifest_done['config_sha256']
manifest_completion_sha = sha(manifest / 'completed.json')

run_done, run_config, checkpoints = {}, {}, {}
for arm, directory in run_dirs.items():
    done = json.loads((directory / 'completed.json').read_text())
    config = json.loads((directory / 'config.json').read_text())
    checkpoint = directory / 'joint_step_06000.pt'
    assert sha(checkpoint) == done['checkpoint_sha256']['6000']
    assert sha(directory / 'config.json') == done['config_sha256']
    assert sha(directory / 'draws.jsonl') == done['draws_sha256']
    assert done['draws_sha256'] == manifest_done['draws_sha256']
    assert done['manifest_updates_sha256'] == manifest_done['updates_sha256']
    assert done['manifest_completion_sha256'] == manifest_completion_sha
    assert config['manifest_completion_sha256'] == manifest_completion_sha
    assert config['manifest_updates_sha256'] == manifest_done['updates_sha256']
    assert config['draws_sha256'] == manifest_done['draws_sha256']
    assert config['parent_checkpoint_sha256'] == sha(parent_path)
    assert config['protocol_sha256'] == sha(protocol)
    assert sha(directory / 'training.jsonl') == done['training_sha256']
    assert done['parent_checkpoint_sha256'] == sha(parent_path)
    assert done['protocol_sha256'] == sha(protocol)
    assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
    assert not any(done.get(key, False) or config.get(key, False) for key in
        ('calibrated', 'expert_real_truth_used', 'final_animals_used',
         'public_benchmark_used', 'external_pretrained_weights_used'))
    run_done[arm], run_config[arm], checkpoints[arm] = done, config, checkpoint
assert run_done['control']['draws_sha256'] == run_done['treatment']['draws_sha256']
assert run_done['control']['manifest_updates_sha256'] == \
    run_done['treatment']['manifest_updates_sha256']
assert run_done['control']['manifest_completion_sha256'] == \
    run_done['treatment']['manifest_completion_sha256']
assert run_config['control']['synthetic_provenance'] == run_config['treatment']['synthetic_provenance']

panel_done = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert panel_protocol['seed'] == 2026101012801
assert panel_protocol['augmentation_seed_prefix'] == 2026101012802
assert panel_protocol['in_path_128_protocol_sha256'] == sha(protocol)
assert panel_protocol['in_path_128_design_sha256'] == sha(design)
assert panel_protocol['source_plan_completed_sha256'] == \
    sha(root / 'data/v3_pose_capture_confirmation_plans_001/completed.json')
assert all(sha(panel / 'source' / name) == digest
           for name, digest in panel_protocol['source_sha256'].items())
assert all(sha(root / 'data' / name / 'records.jsonl') == digest
           for name, digest in panel_protocol['prior_panel_records_sha256'].items())
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = sorted((row for row in records if row['eligible']),
                  key=lambda row: (row['synthetic_subject_plan_id'], row['section_id']))
plans = sorted({row['synthetic_subject_plan_id'] for row in records})
assert len(records) == panel_done['physical_sections'] == 256 and len(plans) == 8
assert len(eligible) == panel_done['eligible'] and panel_done['ineligible'] == 256 - len(eligible)
assert all(sha(panel / row['file']) == row['sha256'] and
           row['provenance']['split'] == 'development' for row in records)
assert all(sum(row['synthetic_subject_plan_id'] == plan for row in records) == 32
           for plan in plans)
assert len({row['panel_physical_section_id'] for row in records}) == 256
assert len({row['section_id'] for row in records}) == 256
train_draws = [json.loads(line) for line in (manifest / 'draws.jsonl').open()]
assert all(row['base_lineage']['split'] == 'train' for row in train_draws)
assert not {row['panel_physical_section_id'] for row in records} & {
    row['physical_section_id'] for row in train_draws}
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in records} & {
        row['base_lineage'][key] for row in train_draws}
by_plan = {plan: [row for row in eligible if row['synthetic_subject_plan_id'] == plan]
           for plan in plans}
swap_record = {record['section_id']: by_plan[plans[(index + 1) % 8]][position %
    len(by_plan[plans[(index + 1) % 8]])] for index, plan in enumerate(plans)
    for position, record in enumerate(by_plan[plan])}

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
    run_config['control']['synthetic_provenance']
atlas = context['atlas']
models = {}
for arm, checkpoint_path in {'frozen122': parent_path, **checkpoints}.items():
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
    attach_global_plane_matcher(model, enabled=True)
    assert checkpoint['step'] == (2000 if arm == 'frozen122' else 6000)
    assert checkpoint['config'] == (parent_config if arm == 'frozen122' else run_config[arm])
    assert checkpoint['calibrated'] is False
    model.load_state_dict(checkpoint['model'], strict=True)
    models[arm] = model
    del checkpoint

offsets = torch.linspace(-31.25, 31.25, 9, device='cuda')[None]
weights = torch.tensor([[1., 2., 2., 2., 2., 2., 2., 2., 1.]], device='cuda') / 16
axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
five_pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                            [127.5, 127.5]], device='cuda')
five_chart = five_pixels / side


def beam_for(prediction):
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
    return prior, beam


def original_costs(model, prediction, valid, centre, target_rigid):
    mapped_cost, rigid_cost = [], []
    for ids in torch.arange(160, device='cuda').split(16):
        mode, reflection = (ids // 2)[None], (ids % 2)[None]
        mapped = model.map(prediction, offsets, mode, reflection,
            (map_side, map_side), atlas, weights, source_shape=(side, side))
        surface = F.interpolate(mapped['centre_surface_ccf_ap_dv_ml_um'][0]
            .permute(0, 3, 1, 2), (side, side), mode='bilinear', align_corners=False)
        mapped_cost.extend(((surface.permute(0, 2, 3, 1) - centre[0])
            .norm(dim=-1)[:, valid[0]].mean(-1) / 1000).tolist())
        points = rigid_points_090(prediction['state'][:, mode[0]], reflection, chart)[0]
        rigid_cost.extend(((points - target_rigid).norm(dim=-1)
            [:, valid[0].flatten()].mean(-1) / 1000).tolist())
    return mapped_cost, rigid_cost


def corrected_costs(model, prediction, reflection, match, valid, centre, target_rigid):
    mapped = model.map({**prediction, 'state': match['state']}, offsets,
        torch.arange(16, device='cuda')[None], reflection,
        (map_side, map_side), atlas, weights, source_shape=(side, side))
    surface = F.interpolate(mapped['centre_surface_ccf_ap_dv_ml_um'][0]
        .permute(0, 3, 1, 2), (side, side), mode='bilinear', align_corners=False)
    mapped_cost = ((surface.permute(0, 2, 3, 1) - centre[0])
        .norm(dim=-1)[:, valid[0]].mean(-1) / 1000).tolist()
    points = rigid_points_090(match['state'], reflection, chart)[0]
    rigid_cost = ((points - target_rigid).norm(dim=-1)
        [:, valid[0].flatten()].mean(-1) / 1000).tolist()
    return mapped_cost, rigid_cost


def selected(match, beam, original_mapped, original_rigid,
             corrected_mapped, corrected_rigid):
    score = torch.stack((match['input_score'], match['score']), -1)
    slot, action = divmod(int(score.flatten().argmax()), 2)
    branch = int(beam[0, slot])
    return {'slot': slot, 'branch_id': branch,
        'action': 'original' if action == 0 else 'corrected',
        'mapped_mm': original_mapped[branch] if action == 0 else corrected_mapped[slot],
        'rigid_mm': original_rigid[branch] if action == 0 else corrected_rigid[slot]}


def matched_pair(beam_cost, atlas_support):
    beam_cost = np.asarray(beam_cost)
    best = int(beam_cost.argmin())
    pair = {'best_slot': best, 'best_mapped_mm': float(beam_cost[best]),
            'near_candidate': bool(beam_cost[best] <= 1.5),
            'status': 'no_near_blind_candidate'}
    if not pair['near_candidate']:
        return pair
    support = atlas_support[0] > .5
    overlap = (support & support[best]).sum((-2, -1)).float()
    dice = 2 * overlap / (support.sum((-2, -1)) + support[best].sum()).clamp_min(1)
    area_gap = (support.float().mean((-2, -1))
                - support[best].float().mean()).abs()
    wrong = ((torch.as_tensor(beam_cost, device='cuda') > beam_cost[best] + 1.)
             & (dice >= .85) & (area_gap <= .08) & (overlap >= 32))
    pair['support_matched_wrong_count'] = int(wrong.sum())
    pair['status'] = 'no_support_area_common_matched_wrong'
    if bool(wrong.any()):
        wrong_slot = int(torch.as_tensor(beam_cost, device='cuda')
            .masked_fill(~wrong, torch.inf).argmin())
        pair.update(status='scored', wrong_slot=wrong_slot,
            wrong_mapped_mm=float(beam_cost[wrong_slot]),
            support_dice=float(dice[wrong_slot]),
            support_area_gap=float(area_gap[wrong_slot]),
            common_support_pixels=int(overlap[wrong_slot]))
    return pair


def pair_margin(match, pair):
    if pair['status'] != 'scored':
        return None
    scores = torch.maximum(match['input_score'], match['score'])[0]
    return float(scores[pair['best_slot']] - scores[pair['wrong_slot']])


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
        target_rigid = rigid_points_090(truth, truth_reflection, chart)[0]
        row = {'section_id': record['section_id'],
            'physical_section_id': record['panel_physical_section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'appearance_mode': record['appearance_mode'],
            'nearest_cardinal_angle_deg': float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0, 1)))),
            'valid_tissue_fraction': float(valid.float().mean()),
            'panel_file_sha256': record['sha256'],
            'swap_section_id': swap_record[record['section_id']]['section_id'],
            'arms': {}, 'treatment_ablations': {}}
        treatment_state = None
        for arm in arm_names:
            model = models[arm]
            prediction = model.predict(image)
            prior, beam = beam_for(prediction)
            mode, reflection = beam // 2, beam % 2
            original_mapped, original_rigid = original_costs(
                model, prediction, valid, centre, target_rigid)
            match = (global_plane_match(model, prediction, mode, reflection,
                offsets, weights, atlas, (side, side), side=24)
                if arm == 'frozen122' else
                global_plane_match_128(model, prediction, mode, reflection,
                offsets, weights, atlas, (side, side), side=24, return_support=True))
            corrected_mapped, corrected_rigid = corrected_costs(
                model, prediction, reflection, match, valid, centre, target_rigid)
            choice = selected(match, beam, original_mapped, original_rigid,
                              corrected_mapped, corrected_rigid)
            beam_ids = beam[0].tolist()
            prior_ids = prior[0].argsort(descending=True).tolist()
            best = {'all160_original': {'mapped_mm': min(original_mapped),
                                       'rigid_mm': min(original_rigid)},
                    'blind16_original': {'mapped_mm': min(original_mapped[i] for i in beam_ids),
                                         'rigid_mm': min(original_rigid[i] for i in beam_ids)},
                    'blind16_corrected': {'mapped_mm': min(corrected_mapped),
                                          'rigid_mm': min(corrected_rigid)}}
            best.update({f'prior_top{count}_original': {
                'mapped_mm': min(original_mapped[i] for i in prior_ids[:count]),
                'rigid_mm': min(original_rigid[i] for i in prior_ids[:count])}
                for count in (1, 16, 32, 64)})
            arm_row = {'beam_branch_ids': beam_ids,
                'prior_ranked_branch_ids': prior_ids, 'selected': choice, 'best': best,
                'original_mapped_mm_by_branch_id': original_mapped,
                'original_rigid_mm_by_branch_id': original_rigid,
                'corrected_mapped_mm_by_slot': corrected_mapped,
                'corrected_rigid_mm_by_slot': corrected_rigid,
                'input_scores': match['input_score'][0].tolist(),
                'corrected_scores': match['score'][0].tolist()}
            if arm != 'frozen122':
                original_beam_cost = np.asarray(original_mapped)[beam_ids]
                action_beam_cost = np.minimum(original_beam_cost,
                                              np.asarray(corrected_mapped))
                pair = matched_pair(action_beam_cost, match['atlas_support'])
                pair['margin'] = pair_margin(match, pair)
                original_pair = matched_pair(original_beam_cost, match['atlas_support'])
                original_pair['margin'] = pair_margin(match, original_pair)
                arm_row['pair'] = pair
                arm_row['original_only_pair_sensitivity'] = original_pair
            row['arms'][arm] = arm_row
            if arm == 'treatment':
                treatment_state = (prediction, beam, mode, reflection, original_mapped,
                                   original_rigid, pair, original_pair)
        prediction, beam, mode, reflection, original_mapped, original_rigid, pair, original_pair = treatment_state
        model = models['treatment']
        shuffled = model.predict(swapped_image)
        shuffle_seed = int.from_bytes(hashlib.sha256(record['section_id'].encode()).digest()[:8],
                                      'little') % (2 ** 63)
        row['shuffle_seed'] = shuffle_seed
        for ablation in ablation_names:
            perturbed = ({**prediction, 'feature': shuffled['feature']}
                         if ablation == 'source_swap' else prediction)
            match = global_plane_match_128(model, perturbed, mode, reflection,
                offsets, weights, atlas, (side, side), side=24,
                ablation=None if ablation == 'source_swap' else ablation,
                shuffle_seed=shuffle_seed)
            corrected_mapped, corrected_rigid = corrected_costs(
                model, prediction, reflection, match, valid, centre, target_rigid)
            row['treatment_ablations'][ablation] = {
                'selected': selected(match, beam, original_mapped, original_rigid,
                                     corrected_mapped, corrected_rigid),
                'pair_margin': pair_margin(match, pair),
                'original_only_pair_margin': pair_margin(match, original_pair),
                'input_scores': match['input_score'][0].tolist(),
                'corrected_scores': match['score'][0].tolist()}
        synthetic_rows.append(row)
        if len(synthetic_rows) % 32 == 0:
            print(json.dumps({'event': 'synthetic_sections', 'completed': len(synthetic_rows),
                              'eligible': len(eligible)}), flush=True)

    for family, family_records in (('coronal', coronal_records),
                                   ('sagittal', sagittal_records)):
        for record in family_records:
            index = record['array_row_index']
            if family == 'coronal':
                native = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
                image = F.interpolate(torch.from_numpy(native).cuda(), (side, side),
                    mode='bilinear', align_corners=False)
                affine = torch.as_tensor(coronal_affines[index], device='cuda',
                                         dtype=torch.float32)
                old_pixels = (five_pixels + .5) * (192 / side) - .5
                reference = affine[:, 2] + (
                    old_pixels[:, :1] * affine[:, 0] +
                    old_pixels[:, 1:] * affine[:, 1])
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
            chosen = {}
            for arm in arm_names:
                model = models[arm]
                prediction = model.predict(image)
                _, beam = beam_for(prediction)
                mode, reflection = beam // 2, beam % 2
                match = (global_plane_match(model, prediction, mode, reflection,
                    offsets, weights, atlas, (side, side), side=24)
                    if arm == 'frozen122' else
                    global_plane_match_128(model, prediction, mode, reflection,
                    offsets, weights, atlas, (side, side), side=24))
                slot, action = divmod(int(torch.stack((
                    match['input_score'], match['score']), -1).flatten().argmax()), 2)
                state = (match['input_state'] if action == 0 else match['state'])[:, slot]
                proposed = rigid_points_090(state, reflection[:, slot], five_chart)[0]
                chosen[arm] = {'slot': slot, 'branch_id': int(beam[0, slot]),
                    'action': 'original' if action == 0 else 'corrected',
                    'weak_five_point_mm': float((proposed - reference).norm(dim=-1).mean() / 1000)}
            real_rows.append({'family': family, 'donor_id': donor,
                'section_id': record['section_id'], 'selected': chosen,
                'label_role': 'inherited weak Allen affine only; not expert truth'})
        print(json.dumps({'event': 'weak_real_family', 'family': family,
                          'sections': len(family_records)}), flush=True)

assert len(synthetic_rows) == len(eligible)
assert len(real_rows) == len(coronal_records) + len(sagittal_records) == 222


def summarise(group):
    result = {'sections': len(group), 'arms': {}, 'treatment_ablations': {}}
    keys = ('prior_top1_original', 'prior_top16_original', 'prior_top32_original',
            'prior_top64_original', 'all160_original', 'blind16_original',
            'blind16_corrected', 'selected')
    for arm in arm_names:
        result['arms'][arm] = {
            'mean_mm': {key: {metric: float(np.mean([
                (row['arms'][arm]['selected'] if key == 'selected'
                 else row['arms'][arm]['best'][key])[metric] for row in group]))
                if group else None for metric in ('mapped_mm', 'rigid_mm')}
                for key in keys},
            'mapped_capture_le_1p5': {key: float(np.mean([
                (row['arms'][arm]['selected'] if key == 'selected'
                 else row['arms'][arm]['best'][key])['mapped_mm'] <= 1.5 for row in group]))
                if group else None for key in keys},
            'selected_action_counts': {action: sum(
                row['arms'][arm]['selected']['action'] == action for row in group)
                for action in ('original', 'corrected')}}
        if arm != 'frozen122':
            pairs = [row['arms'][arm]['pair'] for row in group]
            scored = [pair for pair in pairs if pair['status'] == 'scored']
            result['arms'][arm]['pair'] = {
                'status_counts': {status: sum(pair['status'] == status for pair in pairs)
                    for status in sorted({pair['status'] for pair in pairs})},
                'scored': len(scored),
                'win_fraction': float(np.mean([pair['margin'] >= .01 for pair in scored]))
                    if scored else None,
                'mean_margin': float(np.mean([pair['margin'] for pair in scored]))
                    if scored else None}
            original_pairs = [row['arms'][arm]['original_only_pair_sensitivity']
                              for row in group]
            original_scored = [pair for pair in original_pairs
                               if pair['status'] == 'scored']
            result['arms'][arm]['original_only_pair_sensitivity'] = {
                'status_counts': {status: sum(pair['status'] == status
                    for pair in original_pairs)
                    for status in sorted({pair['status'] for pair in original_pairs})},
                'scored': len(original_scored),
                'win_fraction': float(np.mean([
                    pair['margin'] >= .01 for pair in original_scored]))
                    if original_scored else None}
    treatment_scored = [row for row in group
                        if row['arms']['treatment']['pair']['status'] == 'scored']
    original_scored = [row for row in group if row['arms']['treatment']
                       ['original_only_pair_sensitivity']['status'] == 'scored']
    for ablation in ablation_names:
        result['treatment_ablations'][ablation] = {
            'selected_mapped_mm': float(np.mean([
                row['treatment_ablations'][ablation]['selected']['mapped_mm']
                for row in group])) if group else None,
            'selected_rigid_mm': float(np.mean([
                row['treatment_ablations'][ablation]['selected']['rigid_mm']
                for row in group])) if group else None,
            'selected_mapped_capture_le_1p5': float(np.mean([
                row['treatment_ablations'][ablation]['selected']['mapped_mm'] <= 1.5
                for row in group])) if group else None,
            'same_treatment_pair_win_fraction': float(np.mean([
                row['treatment_ablations'][ablation]['pair_margin'] >= .01
                for row in treatment_scored])) if treatment_scored else None,
            'same_treatment_scored_pairs': len(treatment_scored),
            'original_only_same_pair_win_fraction': float(np.mean([
                row['treatment_ablations'][ablation]['original_only_pair_margin'] >= .01
                for row in original_scored])) if original_scored else None,
            'original_only_same_pair_scored': len(original_scored)}
    return result


by_plan_rows = {plan: [row for row in synthetic_rows
                       if row['synthetic_subject_plan_id'] == plan] for plan in plans}
by_mode_rows = {mode: [row for row in synthetic_rows
                       if row['appearance_mode'] == mode]
                for mode in ('raw', 'exact_black', 'imperfect_brush')}
by_angle_rows = {f'[{low},{high})': [row for row in synthetic_rows
    if low <= row['nearest_cardinal_angle_deg'] < high]
    for low, high in ((0, 15), (15, 30), (30, 45), (45, 55))}
synthetic_summary = {'eligible_sections': len(eligible),
    'panel_physical_sections': len(records),
    'panel_ineligible_sections': panel_done['ineligible'],
    'section_weighted': summarise(synthetic_rows),
    'by_plan': {plan: summarise(group) for plan, group in by_plan_rows.items()},
    'by_mode': {mode: summarise(group) for mode, group in by_mode_rows.items()},
    'by_nearest_cardinal_angle_deg': {name: summarise(group)
        for name, group in by_angle_rows.items()},
    'plan_equal': {'arms': {}, 'treatment_ablations': {}}}
for arm in arm_names:
    summaries = [synthetic_summary['by_plan'][plan]['arms'][arm] for plan in plans]
    synthetic_summary['plan_equal']['arms'][arm] = {
        'selected_mapped_mm': float(np.mean([
            item['mean_mm']['selected']['mapped_mm'] for item in summaries])),
        'selected_rigid_mm': float(np.mean([
            item['mean_mm']['selected']['rigid_mm'] for item in summaries])),
        'selected_mapped_capture_le_1p5': float(np.mean([
            item['mapped_capture_le_1p5']['selected'] for item in summaries])),
        'blind16_original_mapped_capture_le_1p5': float(np.mean([
            item['mapped_capture_le_1p5']['blind16_original'] for item in summaries])),
        'all160_original_mapped_capture_le_1p5': float(np.mean([
            item['mapped_capture_le_1p5']['all160_original'] for item in summaries]))}
for ablation in ablation_names:
    synthetic_summary['plan_equal']['treatment_ablations'][ablation] = {
        'selected_mapped_mm': float(np.mean([
            synthetic_summary['by_plan'][plan]['treatment_ablations'][ablation]
                ['selected_mapped_mm'] for plan in plans])),
        'selected_mapped_capture_le_1p5': float(np.mean([
            synthetic_summary['by_plan'][plan]['treatment_ablations'][ablation]
                ['selected_mapped_capture_le_1p5'] for plan in plans]))}

real_summary = {}
for family in ('coronal', 'sagittal'):
    family_rows = [row for row in real_rows if row['family'] == family]
    donors = sorted({row['donor_id'] for row in family_rows})
    donor_rows = {str(donor): [row for row in family_rows if row['donor_id'] == donor]
                  for donor in donors}
    donor_mean = {donor: {'sections': len(group),
        'selected_weak_five_point_mm': {arm: float(np.mean([
            row['selected'][arm]['weak_five_point_mm'] for row in group]))
            for arm in arm_names}} for donor, group in donor_rows.items()}
    real_summary[family] = {'sections': len(family_rows), 'donors': len(donors),
        'by_donor': donor_mean,
        'donor_equal_weak_five_point_mm': {arm: float(np.mean([
            value['selected_weak_five_point_mm'][arm] for value in donor_mean.values()]))
            for arm in arm_names},
        'label_role': 'inherited weak Allen affine only; not expert truth'}

plan_equal = synthetic_summary['plan_equal']['arms']
pair = synthetic_summary['section_weighted']['arms']['treatment']['pair']
ablation_pair = synthetic_summary['section_weighted']['treatment_ablations']
conditions = {
    'selected_mapped_gain_ge_0p30_vs_matched_control':
        plan_equal['control']['selected_mapped_mm']
        - plan_equal['treatment']['selected_mapped_mm'] >= .30,
    'selected_mapped_capture_gain_ge_0p10_vs_matched_control':
        plan_equal['treatment']['selected_mapped_capture_le_1p5']
        - plan_equal['control']['selected_mapped_capture_le_1p5'] >= .10,
    'plans_not_worse_ge_6_of_8': sum(
        synthetic_summary['by_plan'][plan]['arms']['treatment']['mean_mm']['selected']['mapped_mm']
        <= synthetic_summary['by_plan'][plan]['arms']['control']['mean_mm']['selected']['mapped_mm']
        for plan in plans) >= 6,
    'selected_rigid_no_regression_vs_control':
        plan_equal['treatment']['selected_rigid_mm']
        <= plan_equal['control']['selected_rigid_mm'],
    'blind16_original_capture_drop_le_0p05_vs_control':
        plan_equal['treatment']['blind16_original_mapped_capture_le_1p5']
        >= plan_equal['control']['blind16_original_mapped_capture_le_1p5'] - .05,
    'all160_original_capture_drop_le_0p05_vs_control':
        plan_equal['treatment']['all160_original_mapped_capture_le_1p5']
        >= plan_equal['control']['all160_original_mapped_capture_le_1p5'] - .05,
    'raw_no_brush_selected_mapped_no_regression_vs_control':
        synthetic_summary['by_mode']['raw']['arms']['treatment']['mean_mm']['selected']['mapped_mm']
        <= synthetic_summary['by_mode']['raw']['arms']['control']['mean_mm']['selected']['mapped_mm'],
    'support_matched_pairs_ge_64': pair['scored'] >= 64,
    'pair_win_over_atlas_intensity_zero_ge_0p10':
        pair['win_fraction'] is not None and
        ablation_pair['atlas_intensity_zero']['same_treatment_pair_win_fraction'] is not None and
        pair['win_fraction'] -
        ablation_pair['atlas_intensity_zero']['same_treatment_pair_win_fraction'] >= .10,
    'pair_win_over_source_swap_ge_0p10':
        pair['win_fraction'] is not None and
        ablation_pair['source_swap']['same_treatment_pair_win_fraction'] is not None and
        pair['win_fraction'] -
        ablation_pair['source_swap']['same_treatment_pair_win_fraction'] >= .10,
    'pair_win_over_geometry_only_ge_0p10':
        pair['win_fraction'] is not None and
        ablation_pair['geometry_only']['same_treatment_pair_win_fraction'] is not None and
        pair['win_fraction'] -
        ablation_pair['geometry_only']['same_treatment_pair_win_fraction'] >= .10}
for family in ('coronal', 'sagittal'):
    real = real_summary[family]
    conditions[f'{family}_donor_equal_regression_le_0p20_vs_both'] = all(
        real['donor_equal_weak_five_point_mm']['treatment'] <=
        real['donor_equal_weak_five_point_mm'][reference] + .20
        for reference in ('frozen122', 'control'))
    conditions[f'{family}_each_donor_regression_le_0p50_vs_both'] = all(
        donor['selected_weak_five_point_mm']['treatment'] <=
        donor['selected_weak_five_point_mm'][reference] + .50
        for donor in real['by_donor'].values()
        for reference in ('frozen122', 'control'))

summary = {'scope': 'fresh synthetic sections on reused eight DEV deformation plans; weak-real Allen-affine guards only',
    'mapped_metric': 'valid-original-pixel mean 3D tissue-coordinate error after 96-grid atlas-conditioned mapping and bilinear 256-grid upsampling',
    'rigid_metric': 'same valid-original-pixel mean 3D affine-gauge error without local mapping',
    'pair_definition': 'for each blind16 branch, mapped cost=min(original, corrected) valid-pixel error; near=lowest branch <=1.5 mm; wrong branch >near+1 mm, original-plane 24-grid atlas-support Dice >=.85, area gap <=.08, common support >=32 pixels; nearest eligible wrong; max(original, corrected) score per branch, margin >=.01 wins',
    'original_only_pair_sensitivity': 'same support rules and score, but near/wrong labels use original-action mapped cost only; paired as a separate sensitivity, not the primary gate',
    'ablation_scope': 'conditional matcher perturbations on fixed default-treatment near/wrong branch pair and candidate states/support; ablated correction may change physical quality, so fixed-pair margins test evidence reliance but not counterfactual near/wrong truth or whole-pipeline image independence',
    'synthetic': synthetic_summary, 'real_weak_affine': real_summary,
    'advance_gate': {'conditions': conditions, 'passed': all(conditions.values()),
        'development_only': True},
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
config = {'protocol_sha256': sha(protocol), 'design_sha256': sha(design),
    'evaluator_source_sha256': sha(__file__),
    'matcher_128_source_sha256': sha(source / 'in_path_global_matcher_128.py'),
    'matcher_120_source_sha256': sha(source / 'global_plane_matcher_120.py'),
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'arm_completion_sha256': {arm: sha(directory / 'completed.json')
        for arm, directory in run_dirs.items()},
    'arm_config_sha256': {arm: sha(directory / 'config.json')
        for arm, directory in run_dirs.items()},
    'arm_checkpoint_sha256': {arm: sha(checkpoints[arm]) for arm in run_dirs},
    'matched_draws_sha256': run_done['control']['draws_sha256'],
    'manifest_completion_sha256': manifest_completion_sha,
    'manifest_updates_sha256': manifest_done['updates_sha256'],
    'manifest_config_sha256': manifest_done['config_sha256'],
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'coronal_completion_sha256': sha(coronal / 'completed.json'),
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'sagittal_train_summary_sha256': sha(sagittal_train / 'summary.json'),
    'input_resolution': side, 'map_resolution': map_side,
    'fixed_psf_um': 62.5, 'psf_samples': 9,
    'eligible_synthetic_sections': len(eligible),
    'real_coronal_sections': len(coronal_records),
    'real_sagittal_sections': len(sagittal_records),
    'coronal_weak_affine_resize': 'align_corners=False: native192 pixel=(model256 pixel+0.5)*192/256-0.5',
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for name, rows in (('synthetic_rows.jsonl', synthetic_rows),
                   ('real_weak_rows.jsonl', real_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({
    'eligible_synthetic_sections': len(eligible),
    'real_coronal_sections': len(coronal_records),
    'real_sagittal_sections': len(sagittal_records),
    'support_matched_pairs': pair['scored'],
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'summary.json', 'synthetic_rows.jsonl', 'real_weak_rows.jsonl')},
    'protocol_sha256': config['protocol_sha256'],
    'evaluator_source_sha256': config['evaluator_source_sha256'],
    'arm_checkpoint_sha256': config['arm_checkpoint_sha256'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'eligible_synthetic_sections': len(eligible),
    'support_matched_pairs': pair['scored'], 'advance_gate': summary['advance_gate']},
    allow_nan=False), flush=True)
