"""Use slice-to-atlas matches to fit, warp, and rescore pose candidates."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.atlas_oriented_patch_025 import extract_patches
from training.pose_fine_patch_045 import atlas_key_patches


def joint_pose_fit(pose, coarse, fine, prediction, image, atlas, offsets, weights,
                   beam=4, shortlist=16, fit_side=64):
    batch = len(image)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = prior.topk(beam, -1).indices
    mode, reflection = choice // 2, choice % 2
    state = prediction['state'].gather(1, mode[..., None].expand(-1, -1, 12))
    with torch.no_grad():
        detached = {key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in prediction.items()}
        coarse_result = coarse(detached, image, state.detach(), reflection, atlas,
                               offsets, weights)
        visibility = coarse_result['visibility_logit']
        positions = torch.arange(256, device=image.device).reshape(16, 16)
        query = torch.cat([positions[y:y + 8, x:x + 8].flatten()[
            visibility[:, positions[y:y + 8, x:x + 8].flatten()].topk(2, -1).indices]
            for y in (0, 8) for x in (0, 8)], -1)
        logits = coarse_result['logits'].gather(2, query[:, None, :, None].expand(
            batch, beam, 8, coarse_result['logits'].shape[-1]))
        coarse_score, key_index = logits.topk(shortlist, -1)
        support = coarse_result['key_support'][:, :, None].expand(
            -1, -1, 8, -1).gather(-1, key_index) > .1
        query_weight = visibility.gather(1, query).sigmoid()
    centre, frame, basis = full_frame_state_to_components(state)
    edges = frame[..., :, :2] @ basis
    depth = (key_index // 576).to(state.dtype) * 1000 - 6000
    col = (key_index % 24).to(state.dtype)
    row = ((key_index // 24) % 24).to(state.dtype)
    ax, ay = (col + .5) / 12 - .5, (row + .5) / 12 - .5
    ax = torch.where(reflection[..., None, None].bool(), 255 / 256 - ax, ax)
    key_chart = torch.stack((ax, ay), -1)
    world = (centre[:, :, None, None] +
             torch.einsum('bkij,bkqsj->bkqsi', edges, key_chart - .5) +
             depth[..., None] * frame[:, :, None, None, :, 2])
    qx = ((query % 16).to(state.dtype) + .5) / 16
    qy = ((query // 16).to(state.dtype) + .5) / 16
    qx = torch.where(reflection[..., None].bool(), 255 / 256 - qx[:, None], qx[:, None])
    chart = torch.stack((qx.expand(-1, beam, -1), qy[:, None].expand(-1, beam, -1)), -1)
    query_world = centre[:, :, None] + torch.einsum('bkij,bkqj->bkqi', edges, chart - .5)
    relative = torch.einsum('bkqsi,bkij->bkqsj', world - query_world[..., None, :], frame)
    lists = batch * beam * 8
    yx = torch.stack((8 + 16 * (query // 16), 8 + 16 * (query % 16)), -1)
    yx = yx[:, None].expand(-1, beam, -1, -1).reshape(lists, 2).float()
    image_rows = torch.arange(batch, device=image.device)[:, None, None].expand(
        -1, beam, 8).reshape(-1)
    image_patches = extract_patches(image, image_rows, yx)
    states = state[:, :, None].expand(-1, -1, 8, -1).reshape(lists, 12)
    flags = reflection[:, :, None].expand(-1, -1, 8).reshape(lists)
    z = offsets[:, None, None].expand(-1, beam, 8, -1).reshape(lists, -1)
    w = weights[:, None, None].expand(-1, beam, 8, -1).reshape(lists, -1)
    world = world.reshape(lists, shortlist, 3)
    relative = relative.reshape(lists, shortlist, 3)
    coarse_score = coarse_score.reshape(lists, shortlist)

    def patch_scores(keys, states, flags, z, w, image_patches, relative, coarse_score):
        count = len(keys)
        patches = atlas_key_patches(atlas, keys.reshape(-1, 3),
            states[:, None].expand(-1, shortlist, -1).reshape(-1, 12),
            flags[:, None].expand(-1, shortlist).reshape(-1),
            z[:, None].expand(-1, shortlist, -1).reshape(-1, z.shape[-1]),
            w[:, None].expand(-1, shortlist, -1).reshape(-1, w.shape[-1]))
        return fine(image_patches[:count], patches, relative, coarse_score)[0]

    scores = torch.cat([checkpoint(patch_scores, world[first:first + 2],
        states[first:first + 2], flags[first:first + 2], z[first:first + 2],
        w[first:first + 2], image_patches[first:first + 2],
        relative[first:first + 2], coarse_score[first:first + 2],
        use_reentrant=False) if torch.is_grad_enabled() else patch_scores(
        world[first:first + 2], states[first:first + 2], flags[first:first + 2],
        z[first:first + 2], w[first:first + 2], image_patches[first:first + 2],
        relative[first:first + 2], coarse_score[first:first + 2])
        for first in range(0, lists, 2)], 0).reshape(batch, beam, 8, shortlist)
    world = world.reshape(batch, beam, 8, shortlist, 3)
    support = support & support.any(-1, keepdim=True)
    probability = (scores / 2).masked_fill(~support, -1e4).softmax(-1)
    matched = (probability[..., None] * world).sum(-2)
    confidence = (query_weight[:, None] * probability.amax(-1) *
                  support.any(-1).float()).detach()
    design = torch.cat((torch.ones_like(chart[..., :1]), chart - .5), -1)
    prior_fit = torch.stack((centre, edges[..., :, 0], edges[..., :, 1]), -2)
    ridge = torch.diag(state.new_tensor((2., 1., 1.)))

    def solve(weight):
        lhs = torch.einsum('bkqi,bkq,bkqj->bkij', design, weight, design) + ridge
        rhs = torch.einsum('bkqi,bkq,bkqj->bkij', design, weight, matched) + ridge @ prior_fit
        return torch.linalg.solve(lhs, rhs)

    fitted = solve(confidence)
    residual = (design @ fitted - matched).norm(dim=-1)
    fitted = solve(confidence * (1500 / residual.clamp_min(1500)).detach())
    residual = (design @ fitted - matched).norm(dim=-1)
    residual_mean = (confidence * residual).sum(-1) / confidence.sum(-1).clamp_min(1e-3)
    fitted = prior_fit + .5 * (fitted - prior_fit)
    fit_centre, u, v = fitted.unbind(-2)
    fit_state = full_frame_state_from_components(*physical_ouv_to_frame(
        torch.stack((fit_centre - .5 * (u + v), u, v), -2)))
    selected = {**prediction, 'state': fit_state,
                'log_mass': prediction['log_mass'].gather(1, mode),
                'reflection_logit': prediction['reflection_logit'].gather(1, mode)}
    index = torch.arange(beam, device=image.device)[None].expand(batch, -1)
    mapped = pose.map(selected, offsets, index, reflection, (fit_side, fit_side),
                      atlas, weights, feature_side=fit_side, source_shape=image.shape[-2:])
    fitted_score = pose.score_fitted_candidates(image, selected, mapped, atlas, weights)
    return {'choice': choice, 'prior': prior.gather(1, choice), 'fitted_state': fit_state,
            'mapped': mapped, 'score': fitted_score - .25 * residual_mean / 1000,
            'match_residual_um': residual_mean, 'match_probability': probability}
