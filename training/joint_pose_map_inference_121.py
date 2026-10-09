"""Blind 121 pose/action selection and full-resolution CCF mapping; uncalibrated."""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match


def load_joint_pose_map_121_checkpoint(path, device='cuda'):
    path = Path(path).resolve()
    step = int(path.stem.rsplit('_', 1)[-1])
    if path.name != f'joint_step_{step:05d}.pt':
        raise ValueError('Expected a 121 joint_step checkpoint')
    run = path.parent
    completed = json.loads((run / 'completed.json').read_text())
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text())
    if (completed['updates'] != config['updates'] or config['updates'] != 8000
            or completed['accepted_synthetic'] != 16000
            or completed['fit_feedback_gradient_audited'] is not True
            or step not in config['checkpoints']
            or config['seed'] != 20261009121 or config['side'] != 256
            or config['blind_count'] != 16
            or config['blind_beam'] != '111 prior top 8 base + 6 anchor, plus 2 diverse unused anchors'
            or config['spatial_083_role'] != 'training-only auxiliary correspondence CE; not used by inference map or score'
            or Path(config['parent_checkpoint']).name != 'joint_step_01959.pt'
            or Path(config['parent_checkpoint']).parent.name != 'v3_mixed_real_pose_coronal_risk_111'
            or completed['parent_checkpoint_sha256'] != config['parent_checkpoint_sha256']
            or completed['protocol_sha256'] != config['protocol_sha256']
            or completed['source_sha256'] != config['source_sha256']
            or any(record.get('calibrated') is not False for record in (config, completed))
            or any(record.get('public_benchmark_used') is not False
                   for record in (config, completed))):
        raise ValueError('Checkpoint is not from the completed, uncalibrated 121 lineage')
    for name in ('arbitrary_plane_one_shot_model.py', 'global_plane_matcher_120.py',
                 'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
                 'arbitrary_plane_ribbon_v6.py'):
        with (Path(__file__).resolve().parent / name).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != config['source_sha256'][name]:
            raise RuntimeError(f'121 inference source differs from checkpoint: {name}')
    with config_path.open('rb') as stream:
        config_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    with path.open('rb') as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    if (config_sha256 != completed['config_sha256']
            or checkpoint_sha256 != completed['checkpoint_sha256'][str(step)]):
        raise ValueError('121 checkpoint/config hash differs from completed run')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['step'] != step or checkpoint['config'] != config
            or checkpoint['calibrated'] is not False):
        raise ValueError('121 checkpoint metadata differs from completed run')
    model = OneShotJointSliceModel(
        modes=16, normal_anchor_count=64, atlas_conditioning=True, fit_quality=True,
        vector_refinement=True, candidate_ranking=True, fitted_ranking=True)
    attach_global_plane_matcher(model, enabled=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    model.joint_121_provenance = {
        'checkpoint_path': str(path), 'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_step': step,
        'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
        'protocol_sha256': config['protocol_sha256'],
        'selection_rule': config['blind_beam'],
    }
    return model, config


def infer_joint_pose_map_121(model, inputs, atlas, offsets_um, weights, context=None, chunk=2):
    if (inputs.shape != (1, 5, 256, 256) or model.modes != 80
            or not hasattr(model, 'global_plane_matcher_120')
            or not hasattr(model, 'joint_121_provenance')):
        raise ValueError('121 inference requires one prepared 5×256 image and a loaded 121 model')
    with torch.inference_mode():
        prediction = model.predict(inputs, context)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                          prior[:, 32:].topk(6, -1).indices + 32), -1)
        normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        row = torch.arange(len(inputs), device=inputs.device)
        for _ in range(2):
            chosen = normals[row[:, None], beam // 2]
            similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
            diversity = -similarity
            diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
            anchor = diversity.argmax(-1)
            reflected = prior[:, 32:].reshape(len(inputs), 64, 2)[row, anchor].argmax(-1)
            beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
        mode, reflection = beam // 2, beam % 2
        match = global_plane_match(model, prediction, mode, reflection, offsets_um, weights,
                                   atlas, inputs.shape[-2:], side=24)
        scores = torch.stack((match['input_score'], match['score']), -1)
        choice = int(scores[0].flatten().argmax())
        slot, action = divmod(choice, 2)
        state = (match['input_state'] if action == 0 else match['state'])[:, slot:slot + 1]
        selected = {**prediction, 'state': state}
        mapped = model.map(selected, offsets_um, torch.zeros((1, 1), device=inputs.device,
            dtype=torch.long), reflection[:, slot:slot + 1], inputs.shape[-2:], atlas, weights,
            source_shape=inputs.shape[-2:])
        rendered = render_finite_thickness_coordinate_grid(
            atlas, mapped['coordinates'][0], (0., 0., 0.), (25., 25., 25.), weights)
        candidate_score = torch.full_like(prior, -torch.inf).scatter_(
            1, beam, scores.max(-1).values)
        result = {
            'state': mapped['state'].cpu(),
            'surface': mapped['centre_surface_ccf_ap_dv_ml_um'].cpu(),
            'local_displacement_um': mapped['local_displacement_um'].cpu(),
            'correspondence_logit': mapped['correspondence_logit'].cpu(),
            'atlas_image': (rendered[:, :1] / rendered[:, 1:2].clamp_min(1e-4))
                           .clamp(0, 1).cpu(),
            'prior_log_weight': prior[0].reshape(80, 2).cpu(),
            'candidate_score': candidate_score[0].reshape(80, 2).cpu(),
            'action_score': scores[0].cpu(),
            'beam_branch_ids': beam[0].cpu(),
            'selected_component': (0, 0),
            'source_selected_component': divmod(int(beam[0, slot]), 2),
            'selected_action': 'original' if action == 0 else 'corrected',
            'selected_score': float(scores[0, slot, action]),
            'selection_method': 'joint_121_blind_16_action_score',
            'psf_offsets_um': torch.as_tensor(offsets_um).cpu(),
            'psf_weights': torch.as_tensor(weights).cpu(),
            'provenance': model.joint_121_provenance,
            'calibrated': False,
            'scope': '121 blind 14+2 atlas-conditioned pose/action selection and selected '
                     'full-resolution CCF map; raw uncalibrated scores; no constraints or '
                     'qualified anatomical accuracy',
        }
        return result
