"""All-branch recurrent pose/deformation inference; scores are uncalibrated."""
import torch
import torch.nn.functional as F

from training.arbitrary_plane_joint_model_v8 import JointSliceFeedbackModel


def load_joint_v8_checkpoint(path, device='cuda'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    modes, remainder = divmod(checkpoint['model_state']['pose.2.bias'].numel(), 21)
    if remainder or checkpoint['config'].get('modes', modes) != modes:
        raise ValueError('Checkpoint pose-head mode count is inconsistent')
    model = JointSliceFeedbackModel(modes=modes).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    return model, checkpoint['config']


def infer_joint_v8(model, inputs, atlas, offsets_um, weights, context=None, chunk=2):
    outputs = {name: [] for name in ('state', 'surface', 'residual', 'director', 'joint_std',
                                     'joint_factor', 'mismatch', 'difficulty', 'fit_quality',
                                     'atlas_image', 'pose_sequence', 'pose_updates')}
    with torch.inference_mode():
        prediction = model.predict(inputs, context)
        mode = torch.arange(model.modes, device=inputs.device).repeat_interleave(2)
        reflection = torch.arange(2, device=inputs.device).repeat(model.modes)
        for start in range(0, len(mode), chunk):
            flags = reflection[start:start + chunk]
            fitted = model.fit(prediction, atlas, offsets_um, weights,
                               mode[None, start:start + chunk], flags[None])
            geometry = fitted['geometry']
            surface = geometry['centre_surface_ccf_ap_dv_ml_um']
            surface = torch.where(flags[:, None, None, None].bool(), surface.flip(-2), surface)
            values = (fitted['state'], surface, geometry['residual_local_um'],
                      geometry['director_delta_local'], fitted['joint_std'], fitted['joint_factor'],
                      fitted['mismatch'][0], fitted['difficulty'][0], fitted['fit_quality'][0],
                      fitted['rendered'], fitted['pose_sequence'], fitted['pose_updates'])
            for key, value in zip(outputs, values):
                outputs[key].append(value.cpu())
        result = {key: torch.cat(values).reshape(model.modes, 2, *values[0].shape[1:])
                  for key, values in outputs.items()}
        reflection_lp = torch.stack((F.logsigmoid(-prediction['reflection_logit'][0]),
                                     F.logsigmoid(prediction['reflection_logit'][0])), -1).cpu()
        prior = prediction['log_mass'][0, :, None].cpu() + reflection_lp
        score = result['fit_quality'] + .2 * prior - .05 * result['difficulty']
        result['component_log_weight'] = score - score.flatten().logsumexp(0)
        result['selected_component'] = divmod(int(score.flatten().argmax()), 2)
        result['prior_log_weight'] = prior
        result['calibrated'] = False
        result['scope'] = 'experimental fit-quality ranking, not posterior coverage or region probabilities'
        result['psf_offsets_um'], result['psf_weights'] = offsets_um.cpu(), weights.cpu()
        return result
