"""Frozen 122 all-160 proposal capture on the exact 125 synthetic DEV panel."""

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


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/JOINT_POSE_CAPTURE_127_PROTOCOL_20261010.md'
parent_dir = root / 'runs/joint_pose_map_122'
parent_path = parent_dir / 'joint_step_02000.pt'
panel = root / 'data/joint_dense_anatomy_125_dev_panel'
eval125 = root / 'runs/joint_dense_anatomy_125_dev_eval'
out = root / 'runs/joint_pose_capture_127_diag'
side, map_side = 256, 96
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
parent_done = json.loads((parent_dir / 'completed.json').read_text())
panel_done = json.loads((panel / 'completed.json').read_text())
eval_done = json.loads((eval125 / 'completed.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(eval125 / 'synthetic_rows.jsonl') == eval_done['output_sha256']['synthetic_rows.jsonl']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = sorted((row for row in records if row['eligible']),
                  key=lambda row: (row['synthetic_subject_plan_id'], row['section_id']))
frozen125 = {row['section_id']: row for row in map(json.loads,
             (eval125 / 'synthetic_rows.jsonl').open())}
assert len(records) == panel_done['physical_sections'] == 256
assert len(eligible) == panel_done['eligible'] == len(frozen125) == 241

context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(checkpoint['model'], strict=True)
assert checkpoint['step'] == 2000 and checkpoint['calibrated'] is False
del checkpoint

offsets = torch.linspace(-31.25, 31.25, 9, device='cuda')[None]
weights = torch.tensor([[1., 2., 2., 2., 2., 2., 2., 2., 1.]], device='cuda') / 16
axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)


def costs(prediction, branch_ids, valid, centre, target_rigid):
    mapped_cost, rigid_cost = [], []
    for ids in branch_ids.split(16):
        mode, reflection = (ids // 2)[None], (ids % 2)[None]
        mapped = model.map(prediction, offsets, mode, reflection,
            (map_side, map_side), atlas, weights, source_shape=(side, side))
        surface = F.interpolate(mapped['centre_surface_ccf_ap_dv_ml_um'][0]
            .permute(0, 3, 1, 2), (side, side), mode='bilinear', align_corners=False)
        mapped_cost.extend(((surface.permute(0, 2, 3, 1) - centre[0])
            .norm(dim=-1)[:, valid[0]].mean(-1) / 1000).tolist())
        state = prediction['state'][:, mode[0]]
        points = rigid_points_090(state, reflection, chart)[0]
        rigid_cost.extend(((points - target_rigid).norm(dim=-1)
            [:, valid[0].flatten()].mean(-1) / 1000).tolist())
    return mapped_cost, rigid_cost


rows = []
with torch.inference_mode():
    for record in eligible:
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            truth_reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        order = prior[0].argsort(descending=True)
        branch_ids = torch.arange(160, device='cuda')
        target_rigid = rigid_points_090(truth, truth_reflection, chart)[0]
        mapped160, rigid160 = costs(prediction, branch_ids, valid, centre, target_rigid)

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
        assert beam[0].tolist() == frozen125[record['section_id']]['beam_branch_ids']
        reflection = beam % 2
        match = global_plane_match(model, prediction, beam // 2, reflection,
            offsets, weights, atlas, (side, side), side=24)
        corrected = model.map({**prediction, 'state': match['state']}, offsets,
            torch.arange(16, device='cuda')[None], reflection,
            (map_side, map_side), atlas, weights, source_shape=(side, side))
        corrected_surface = F.interpolate(corrected['centre_surface_ccf_ap_dv_ml_um'][0]
            .permute(0, 3, 1, 2), (side, side), mode='bilinear', align_corners=False)
        corrected_mapped = ((corrected_surface.permute(0, 2, 3, 1) - centre[0])
            .norm(dim=-1)[:, valid[0]].mean(-1) / 1000).tolist()
        corrected_points = rigid_points_090(match['state'], reflection, chart)[0]
        corrected_rigid = ((corrected_points - target_rigid).norm(dim=-1)
            [:, valid[0].flatten()].mean(-1) / 1000).tolist()
        selected_slot, selected_action = divmod(int(torch.stack((
            match['input_score'], match['score']), -1).flatten().argmax()), 2)
        selected_id = int(beam[0, selected_slot])
        selected_mapped = (mapped160[selected_id] if selected_action == 0
                           else corrected_mapped[selected_slot])
        selected_rigid = (rigid160[selected_id] if selected_action == 0
                          else corrected_rigid[selected_slot])
        previous = frozen125[record['section_id']]
        assert selected_slot == previous['selected']['frozen122']['slot']
        assert ('original' if selected_action == 0 else 'corrected') == \
            previous['selected']['frozen122']['action']
        assert abs(min(corrected_mapped) - previous['pair']['best_mapped_mm']) < 1e-4
        assert abs(selected_mapped - previous['selected']['frozen122']['mapped_tissue_mm']) < 1e-4
        assert abs(selected_rigid - previous['selected']['frozen122']['rigid_gauge_mm']) < 1e-4

        top = {str(k): order[:k].tolist() for k in (16, 32, 64)}
        by_id = beam[0].tolist()
        row = {'section_id': record['section_id'],
            'physical_section_id': record['panel_physical_section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'appearance_mode': record['appearance_mode'],
            'nearest_cardinal_angle_deg': float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0, 1)))),
            'valid_tissue_fraction': float(valid.float().mean()),
            'panel_file_sha256': record['sha256'],
            'prior_ranked_branch_ids': order.tolist(),
            'prior_score_by_branch_id': prior[0].tolist(),
            'original_mapped_mm_by_branch_id': mapped160,
            'original_rigid_mm_by_branch_id': rigid160,
            'blind16_branch_ids': by_id,
            'blind16_corrected_mapped_mm_by_slot': corrected_mapped,
            'blind16_corrected_rigid_mm_by_slot': corrected_rigid,
            'selected': {'slot': selected_slot, 'branch_id': selected_id,
                'action': 'original' if selected_action == 0 else 'corrected',
                'mapped_mm': selected_mapped, 'rigid_mm': selected_rigid},
            'best': {'all160_original': {'mapped_mm': min(mapped160),
                'rigid_mm': min(rigid160)},
                'blind16_original': {'mapped_mm': min(mapped160[i] for i in by_id),
                    'rigid_mm': min(rigid160[i] for i in by_id)},
                'blind16_corrected': {'mapped_mm': min(corrected_mapped),
                    'rigid_mm': min(corrected_rigid)}}}
        row['best'].update({f'prior_top{k}_original': {
            'mapped_mm': min(mapped160[i] for i in ids),
            'rigid_mm': min(rigid160[i] for i in ids)} for k, ids in top.items()})
        rows.append(row)
        if len(rows) % 32 == 0:
            print(json.dumps({'event': 'sections', 'completed': len(rows),
                              'eligible': len(eligible)}), flush=True)


def summary(group):
    keys = ('prior_top16_original', 'prior_top32_original', 'prior_top64_original',
            'all160_original', 'blind16_original', 'blind16_corrected', 'selected')
    return {'n': len(group), 'mean_mm': {key: {metric: float(np.mean([
        (row['selected'] if key == 'selected' else row['best'][key])[metric]
        for row in group])) for metric in ('mapped_mm', 'rigid_mm')} for key in keys},
        'mapped_capture_le_1p5': {key: sum((row['selected'] if key == 'selected'
        else row['best'][key])['mapped_mm'] <= 1.5 for row in group) for key in keys},
        'rigid_capture_le_1p5': {key: sum((row['selected'] if key == 'selected'
        else row['best'][key])['rigid_mm'] <= 1.5 for row in group) for key in keys}}


strata = {'appearance_mode': {name: summary([row for row in rows
    if row['appearance_mode'] == name]) for name in ('raw', 'exact_black', 'imperfect_brush')},
    'nearest_cardinal_angle_deg': {f'[{lo},{hi})': summary([row for row in rows
    if lo <= row['nearest_cardinal_angle_deg'] < hi]) for lo, hi in
    ((0, 15), (15, 30), (30, 45), (45, 55))},
    'synthetic_subject_plan_id': {name: summary([row for row in rows
    if row['synthetic_subject_plan_id'] == name]) for name in sorted({
        row['synthetic_subject_plan_id'] for row in rows})}}
overall = summary(rows)
overall['all160_vs_beam16_original_capture_gain_fraction'] = (
    overall['mapped_capture_le_1p5']['all160_original'] -
    overall['mapped_capture_le_1p5']['blind16_original']) / len(rows)
overall['beam_pruning_material_by_predeclared_0p10_rule'] = (
    overall['all160_vs_beam16_original_capture_gain_fraction'] >= .10)
result = {'scope': 'frozen 122 on reused 125 synthetic DEV panel only',
    'original_vs_corrected_note': 'Top-k and all160 use original proposal states; corrected and selected use frozen global matcher only within blind16.',
    'mapped_metric': '256-grid mean 3D tissue-coordinate error over all synthetic-valid original pixels after atlas-conditioned 96-grid mapping and bilinear upsampling',
    'rigid_metric': '256-grid mean 3D affine-gauge point error over the same valid pixels without local mapping',
    'overall': overall, 'strata': strata,
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
config = {'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_joint_pose_capture_127.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
        'global_atlas_contrast_090.py', 'global_plane_matcher_120.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completed_sha256': sha(parent_dir / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'prior_125_synthetic_rows_sha256': sha(eval125 / 'synthetic_rows.jsonl'),
    'synthetic_provenance': context['provenance'],
    'eligible_sections': len(rows), 'mapped_resolution': map_side,
    'input_resolution': side, 'psf_samples': 9, 'fixed_psf_um': 62.5}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(result, indent=2))
(out / 'completed.json').write_text(json.dumps({'eligible_sections': len(rows),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'rows.jsonl', 'summary.json')},
    'protocol_sha256': config['protocol_sha256'],
    'source_sha256': config['source_sha256']['evaluate_joint_pose_capture_127.py'],
    'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'sections': len(rows),
    'all160_vs_beam16_original_capture_gain_fraction':
    overall['all160_vs_beam16_original_capture_gain_fraction'],
    'material_beam_pruning': overall['beam_pruning_material_by_predeclared_0p10_rule']}),
    flush=True)
