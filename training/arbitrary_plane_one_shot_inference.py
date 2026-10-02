"""Full-plane and local-map inference; all scores remain uncalibrated."""
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
        fitted_ranking=any(key.startswith('fitted_matcher.') for key in checkpoint['model']),
        dense_coordinate=any(key.startswith('dense_coordinate_head.') for key in checkpoint['model']),
        vector_refinement=any(key.startswith('pose_refiner.') for key in checkpoint['model']),
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
        mapped_prediction = prediction
        if hasattr(model, 'pose_refiner'):
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                          F.logsigmoid(prediction['reflection_logit'])), -1)
            prior_score = (prediction['log_mass'][..., None] + log_reflection).flatten(1)
            top = prior_score.topk(min(8, 2 * model.modes), -1).indices
            selected = {**prediction,
                        'state': prediction['state'].gather(1, (top // 2)[..., None].expand(-1, -1, 12)),
                        'log_mass': prediction['log_mass'].gather(1, top // 2),
                        'reflection_logit': prediction['reflection_logit'].gather(1, top // 2)}
            index = torch.arange(top.shape[1], device=inputs.device)[None].expand_as(top)
            first = model.map(selected, offsets_um, index, top % 2, (64, 64), atlas, weights,
                              return_refinement_feature=True, feature_side=64,
                              source_shape=inputs.shape[-2:])
            refined_state, delta, _ = model.refine(first['refinement_feature'], selected['state'])
            refined = {**selected, 'state': refined_state}
            fitted_map = model.map(refined, offsets_um, index, top % 2, (96, 96), atlas,
                                   weights, feature_side=96, source_shape=inputs.shape[-2:])
            fitted = model.score_fitted_candidates(inputs, refined, fitted_map, atlas, weights) + delta
            candidate_score = torch.full_like(prior_score, -torch.inf).scatter_(1, top, fitted)
            branch_state = prediction['state'].repeat_interleave(2, dim=1)
            branch_state = branch_state.scatter(1, top[..., None].expand(-1, -1, 12), refined_state)
            mapped_prediction = {**prediction, 'state': branch_state}
        elif hasattr(model, 'fitted_matcher'):
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                            F.logsigmoid(prediction['reflection_logit'])), -1)
            prior_score = (prediction['log_mass'][..., None] + log_reflection).flatten(1)
            top = prior_score.topk(min(8, 2 * model.modes), -1).indices
            coarse = model.map(prediction, offsets_um, top // 2, top % 2, (96, 96),
                               atlas, weights, feature_side=96, source_shape=inputs.shape[-2:])
            fitted = model.score_fitted_candidates(inputs, prediction, coarse, atlas, weights)
            candidate_score = torch.full_like(prior_score, -torch.inf).scatter_(1, top, fitted)
        elif hasattr(model, 'candidate_matcher'):
            candidate_score = model.score_candidates(prediction, offsets_um, weights, atlas,
                                                     use_atlas=model.candidate_uses_atlas,
                                                     side=64, chunk=chunk,
                                                     image_shape=inputs.shape[-2:])
        else:
            candidate_score = None
        for first in range(0, len(mode), chunk):
            chosen = (torch.arange(first, first + min(chunk, len(mode) - first), device=inputs.device)[None]
                      if hasattr(model, 'pose_refiner') else mode[first:first + chunk][None])
            flags = reflection[first:first + chunk][None]
            mapped = model.map(mapped_prediction, offsets_um, chosen, flags,
                               inputs.shape[-2:], atlas, weights)
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
    result['selection_method'] = ('joint_rerender_refinement' if hasattr(model, 'pose_refiner') else
                                  'postfit_atlas_candidate_score' if hasattr(model, 'fitted_matcher') else
                                  'image_prior' if candidate_score is None else
                                  'atlas_candidate_score' if model.candidate_uses_atlas else
                                  'image_candidate_score')
    if candidate_score is not None:
        result['candidate_score'] = candidate_score[0].reshape(model.modes, 2).cpu()
    result['psf_offsets_um'], result['psf_weights'] = offsets_um.cpu(), weights.cpu()
    result['calibrated'] = False
    result['scope'] = ('direct pose and local map; '
                       + result['selection_method'] + '; no calibrated probabilities')
    return result
