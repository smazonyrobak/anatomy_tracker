"""One-pass full-plane and local-map inference; all scores remain uncalibrated."""
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


def load_one_shot_checkpoint(path, device='cuda'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = OneShotJointSliceModel(
        modes=checkpoint['model']['pose.2.bias'].numel() // 21,
        atlas_conditioning=any(key.startswith('atlas_encoder.') for key in checkpoint['model']),
        fit_quality=any(key.startswith('fit_quality_head.') for key in checkpoint['model']),
        candidate_ranking=any(key.startswith('candidate_matcher.') for key in checkpoint['model']),
    ).to(device)
    model.candidate_uses_atlas = checkpoint.get('arm') != 'direct'
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint['config']


def infer_one_shot(model, inputs, atlas, offsets_um, weights, context=None, chunk=2):
    mode = torch.arange(model.modes, device=inputs.device).repeat_interleave(2)
    reflection = torch.arange(2, device=inputs.device).repeat(model.modes)
    output = {name: [] for name in ('state', 'surface', 'local_displacement_um',
                                    'correspondence_logit', 'joint_std', 'joint_factor', 'atlas_image')}
    with torch.inference_mode():
        prediction = model.predict(inputs, context)
        candidate_score = (model.score_candidates(prediction, offsets_um, weights, atlas,
                                                  use_atlas=model.candidate_uses_atlas,
                                                  side=64, chunk=chunk, image_shape=inputs.shape[-2:])
                           if hasattr(model, 'candidate_matcher') else None)
        for first in range(0, len(mode), chunk):
            chosen = mode[first:first + chunk][None]
            flags = reflection[first:first + chunk][None]
            mapped = model.map(prediction, offsets_um, chosen, flags, inputs.shape[-2:], atlas, weights)
            count = chosen.shape[1]
            slab = mapped['coordinates'][0]
            channel_weights = weights.expand(count, -1)
            rendered = render_finite_thickness_coordinate_grid(
                atlas, slab, (0., 0., 0.), (25., 25., 25.), channel_weights)
            values = (mapped['state'][0], mapped['centre_surface_ccf_ap_dv_ml_um'][0],
                      mapped['local_displacement_um'][0], mapped['correspondence_logit'][0],
                      mapped['joint_std'][0], mapped['joint_factor'][0],
                      (rendered[:, 0] / rendered[:, 1].clamp_min(1e-4)).clamp(0, 1))
            for name, value in zip(output, values):
                output[name].append(value.cpu())
    result = {name: torch.cat(values).reshape(model.modes, 2, *values[0].shape[1:])
              for name, values in output.items()}
    prior = prediction['log_mass'][0, :, None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'][0]),
        F.logsigmoid(prediction['reflection_logit'][0])), -1)
    result['prior_log_weight'] = prior.cpu()
    result['component_log_weight'] = prior.cpu()
    selected = prior.flatten() if candidate_score is None else candidate_score[0]
    result['selected_component'] = divmod(int(selected.argmax()), 2)
    result['selection_method'] = ('image_prior' if candidate_score is None else
                                  'atlas_candidate_score' if model.candidate_uses_atlas else
                                  'image_candidate_score')
    if candidate_score is not None:
        result['candidate_score'] = candidate_score[0].reshape(model.modes, 2).cpu()
    result['psf_offsets_um'], result['psf_weights'] = offsets_um.cpu(), weights.cpu()
    result['calibrated'] = False
    result['scope'] = ('one-pass direct pose and local map; '
                       + result['selection_method'] + '; no calibrated probabilities')
    return result
