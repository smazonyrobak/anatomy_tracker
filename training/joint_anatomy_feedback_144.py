"""Direct pose candidates, one tissue map, and differentiable fit feedback."""

import torch
import torch.nn.functional as F

from training.coherent_anatomy_geometry_143 import render_coherent_atlas_143
from training.joint_anatomy_mapping_144 import coherent_surface_144


def joint_anatomy_candidates_144(model, field, inputs, atlas, offsets, weights,
                                 branches=None, count=4, temperature=.25, context=None):
    prediction = model.predict(inputs, context=context)
    batch = len(inputs)
    reflection_mass = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                   F.logsigmoid(prediction['reflection_logit'])), -1)
    prior = (prediction['log_mass'][..., None] + reflection_mass).flatten(1)
    if branches is None:
        branches = prior.topk(count, -1).indices
    count = branches.shape[1]
    row = torch.arange(batch, device=inputs.device)[:, None]
    state = prediction['state'][row, branches // 2]
    reflection = (branches % 2).reshape(-1)
    flat_state = state.flatten(0, 1)
    axial = torch.as_tensor(offsets, device=inputs.device, dtype=inputs.dtype)
    mass = torch.as_tensor(weights, device=inputs.device, dtype=inputs.dtype)
    if axial.ndim == 1:
        axial = axial[None].expand(batch, -1)
    if mass.ndim == 1:
        mass = mass[None].expand(batch, -1)
    axial = axial[:, None].expand(-1, count, -1).flatten(0, 1)
    mass = mass[:, None].expand(-1, count, -1).flatten(0, 1)
    slabs, base, basis, normal = render_coherent_atlas_143(
        atlas, flat_state, reflection, axial, mass)
    source = inputs[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
    feature = prediction['feature'][:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
    match = field(source, slabs, 500., source_feature=feature)
    surface = coherent_surface_144(flat_state, reflection, match['offset_chart'],
                                   inputs.shape[-2:]).reshape(batch, count, *inputs.shape[-2:], 3)
    selected_prior = prior.gather(1, branches)
    energy = match['energy'].reshape(batch, count)
    fit_loss = -temperature * torch.logsumexp(
        selected_prior.log_softmax(-1) - energy / temperature, -1).mean()
    return {'prediction': prediction, 'branches': branches, 'state': state,
            'reflection': reflection.reshape(batch, count), 'surface_ccf_um': surface,
            'local_offset_chart': match['offset_chart'].reshape(batch, count, 3, 64, 64),
            'visibility': match['visibility'].reshape(batch, count, 1, 64, 64),
            'match_confidence_uncalibrated': match['confidence'].reshape(batch, count, 1, 64, 64),
            'prior_log_mass': selected_prior, 'fit_energy_uncalibrated': energy,
            'fit_feedback_loss': fit_loss}
