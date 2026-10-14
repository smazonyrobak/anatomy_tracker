"""Full-plane and local-map inference; all scores remain uncalibrated."""
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


def load_one_shot_checkpoint(path, device='cuda'):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    anchors = checkpoint['model'].get('normal_anchor_frames')
    anchor_hypotheses = (checkpoint['model']['anchor_pose.2.bias'].numel() // (18 * len(anchors))
                         if anchors is not None and 'anchor_pose.2.bias' in checkpoint['model'] else 1)
    model = OneShotJointSliceModel(
        modes=checkpoint['model']['pose.2.bias'].numel() // 21,
        normal_anchor_count=0 if anchors is None else len(anchors),
        anchor_hypotheses=anchor_hypotheses,
        shared_anchor_pose=any(key.startswith('shared_anchor_query.') for key in checkpoint['model']),
        atlas_conditioning=any(key.startswith('atlas_encoder.') for key in checkpoint['model']),
        fit_quality=any(key.startswith('fit_quality_head.') for key in checkpoint['model']),
        candidate_ranking=any(key.startswith('candidate_matcher.') for key in checkpoint['model']),
        fitted_ranking=any(key.startswith('fitted_matcher.') for key in checkpoint['model']),
        dense_coordinate=any(key.startswith('dense_coordinate_head.') for key in checkpoint['model']),
        vector_refinement=any(key.startswith('pose_refiner.') for key in checkpoint['model']),
        spatial_feedback=any(key.startswith('source_reliability_head.') for key in checkpoint['model']),
    ).to(device)
    model.candidate_uses_atlas = checkpoint.get('arm') != 'direct'
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint['config']


def infer_one_shot(model, inputs, atlas, offsets_um, weights, context=None, chunk=2,
                   evidence_arm='full', max_derivative=None):
    mode = torch.arange(model.modes, device=inputs.device).repeat_interleave(2)
    reflection = torch.arange(2, device=inputs.device).repeat(model.modes)
    output = {name: [] for name in ('state', 'surface', 'local_displacement_um',
                                    'correspondence_logit', 'joint_std', 'joint_factor', 'atlas_image')}
    if model.spatial_feedback:
        output['prelimit_local_displacement_um'] = []
    with torch.inference_mode():
        prediction = model.predict(inputs, context)
        mapped_prediction = prediction
        if hasattr(model, 'pose_refiner'):
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                          F.logsigmoid(prediction['reflection_logit'])), -1)
            prior_score = (prediction['log_mass'][..., None] + log_reflection).flatten(1)
            if model.spatial_feedback:
                top = prior_score.topk(min(16, 2 * model.modes), -1).indices
            elif model.shared_anchor_pose:
                top = prior_score.topk(min(8, 2 * model.modes), -1).indices
            elif model.normal_anchor_count:
                old = prior_score[:, :2 * model.base_modes].topk(2, -1).indices
                anchor = (prior_score[:, 2 * model.base_modes:].topk(6, -1).indices
                          + 2 * model.base_modes)
                top = torch.cat((old, anchor), -1)
            else:
                top = prior_score.topk(min(8, 2 * model.modes), -1).indices
            if model.spatial_feedback:
                refined_parts, score_parts = [], []
                for first_index in range(0, top.shape[1], chunk):
                    group = top[:, first_index:first_index + chunk]
                    selected = {**prediction,
                                'state': prediction['state'].gather(
                                    1, (group // 2)[..., None].expand(-1, -1, 12)),
                                'log_mass': prediction['log_mass'].gather(1, group // 2),
                                'reflection_logit': prediction['reflection_logit'].gather(
                                    1, group // 2)}
                    index = torch.arange(group.shape[1], device=inputs.device)[None].expand_as(group)
                    first_map = model.map(
                        selected, offsets_um, index, group % 2, (64, 64), atlas, weights,
                        return_refinement_feature=True, feature_side=64,
                        source_shape=inputs.shape[-2:], max_derivative=max_derivative)
                    refined_state, _, _ = model.refine(first_map['refinement_feature'],
                                                       selected['state'])
                    refined = {**selected, 'state': refined_state}
                    fitted_map = model.map(
                        refined, offsets_um, index, group % 2, (96, 96), atlas, weights,
                        feature_side=96, source_shape=inputs.shape[-2:],
                        max_derivative=max_derivative)
                    score_parts.append(model.score_fitted_candidates(
                        inputs, selected, fitted_map, atlas, weights,
                        evidence_arm=evidence_arm))
                    refined_parts.append(refined_state)
                refined_state = torch.cat(refined_parts, 1)
                fitted = torch.cat(score_parts, 1)
            else:
                selected = {**prediction,
                            'state': prediction['state'].gather(1, (top // 2)[..., None].expand(-1, -1, 12)),
                            'log_mass': prediction['log_mass'].gather(1, top // 2),
                            'reflection_logit': prediction['reflection_logit'].gather(1, top // 2)}
                index = torch.arange(top.shape[1], device=inputs.device)[None].expand_as(top)
                first_map = model.map(selected, offsets_um, index, top % 2, (64, 64), atlas,
                                      weights, return_refinement_feature=True, feature_side=64,
                                      source_shape=inputs.shape[-2:])
                refined_state, delta, _ = model.refine(first_map['refinement_feature'],
                                                       selected['state'])
                refined = {**selected, 'state': refined_state}
                fitted_map = model.map(refined, offsets_um, index, top % 2, (96, 96), atlas,
                                       weights, feature_side=96, source_shape=inputs.shape[-2:])
                fitted = model.score_fitted_candidates(inputs, refined, fitted_map, atlas, weights)
                if not model.shared_anchor_pose:
                    fitted = fitted + delta
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
        compact = bool(model.normal_anchor_count and hasattr(model, 'pose_refiner'))
        branches = (torch.tensor([int(candidate_score[0].argmax())], device=inputs.device)
                    if compact else torch.arange(2 * model.modes, device=inputs.device))
        for first in range(0, len(branches), chunk):
            selected_branches = branches[first:first + chunk]
            chosen = (selected_branches[None] if hasattr(model, 'pose_refiner')
                      else mode[selected_branches][None])
            flags = reflection[selected_branches][None]
            mapped = model.map(mapped_prediction, offsets_um, chosen, flags,
                               inputs.shape[-2:], atlas, weights,
                               max_derivative=max_derivative if model.spatial_feedback else None)
            count = chosen.shape[1]
            slab = mapped['coordinates'][0]
            channel_weights = weights.expand(count, -1)
            rendered = render_finite_thickness_coordinate_grid(
                atlas, slab, (0., 0., 0.), (25., 25., 25.), channel_weights)
            values = (mapped['state'][0], mapped['centre_surface_ccf_ap_dv_ml_um'][0],
                      mapped['local_displacement_um'][0], mapped['correspondence_logit'][0],
                      mapped['joint_std'][0], mapped['joint_factor'][0],
                      (rendered[:, 0] / rendered[:, 1].clamp_min(1e-4)).clamp(0, 1))
            if model.spatial_feedback:
                values += (mapped['prelimit_local_displacement_um'][0],)
            for name, value in zip(output, values):
                output[name].append(value.cpu())
    result = {name: torch.cat(values).reshape(*(1, 1) if compact else (model.modes, 2),
                                              *values[0].shape[1:])
              for name, values in output.items()}
    prior = prediction['log_mass'][0, :, None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'][0]),
        F.logsigmoid(prediction['reflection_logit'][0])), -1)
    result['prior_log_weight'] = prior.cpu()
    result['component_log_weight'] = prior.cpu()
    selected = prior.flatten() if candidate_score is None else candidate_score[0]
    result['selected_component'] = (0, 0) if compact else divmod(int(selected.argmax()), 2)
    if compact:
        result['source_selected_component'] = divmod(int(selected.argmax()), 2)
    result['selection_method'] = ('spatial_feedback_' + evidence_arm if model.spatial_feedback else
                                  'joint_rerender_refinement' if hasattr(model, 'pose_refiner') else
                                  'postfit_atlas_candidate_score' if hasattr(model, 'fitted_matcher') else
                                  'image_prior' if candidate_score is None else
                                  'atlas_candidate_score' if model.candidate_uses_atlas else
                                  'image_candidate_score')
    if candidate_score is not None:
        result['candidate_score'] = candidate_score[0].reshape(model.modes, 2).cpu()
    if model.spatial_feedback:
        result['source_reliability'] = prediction['reliability_logit'].sigmoid().cpu()
        result['max_derivative'] = max_derivative
    result['psf_offsets_um'], result['psf_weights'] = offsets_um.cpu(), weights.cpu()
    result['calibrated'] = False
    result['scope'] = ('direct pose and local map; '
                       + result['selection_method'] + '; no calibrated probabilities')
    return result
