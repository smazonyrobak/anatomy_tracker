"""Seeded full-plane consensus fit for the 237 local 3-D patch matches."""

import torch

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame


def fit_local_3d_consensus(state, reflection, match, seed):
    chart = match['query_chart']
    canonical_x = torch.where(reflection.bool(), 255 / 256 - chart[:, 0], chart[:, 0])
    design = torch.stack((torch.ones_like(canonical_x), canonical_x - .5,
                          chart[:, 1] - .5), -1)
    target = match['matched_world']
    weight = match['confidence'].clamp_min(0) * match['support'].to(chart.dtype)
    centre, frame, basis = full_frame_state_to_components(state)
    edges = frame[0, :, :2] @ basis[0]
    old = torch.stack((centre[0], edges[:, 0], edges[:, 1]))
    anchors = state.new_tensor(((1, -.5, -.5), (1, .5, -.5),
                                (1, -.5, .5), (1, .5, .5), (1, 0, 0)))
    total = weight.sum()
    diagnostics = {'updated': False, 'accepted': False, 'old_score': 0.,
                   'consensus_score': 0., 'final_score': 0.,
                   'five_anchor_movement_um': 0., 'inliers': 0,
                   'valid_triples': 0}
    if int((weight > 0).sum()) < 3:
        return state, diagnostics

    old_residual = torch.linalg.vector_norm(design @ old - target, dim=-1)
    old_score = ((1 - old_residual / 800).clamp_min(0) * weight).sum() / total
    diagnostics['old_score'] = diagnostics['final_score'] = float(old_score)
    generator = torch.Generator(device=state.device).manual_seed(int(seed))
    triples = torch.randint(len(chart), (1536, 3), device=state.device,
                            generator=generator)
    sampled = design[triples]
    valid = (torch.linalg.det(sampled).abs() >= .02) & (weight[triples] > 0).all(-1)
    safe = torch.where(valid[:, None, None], sampled,
                       torch.eye(3, device=state.device, dtype=state.dtype))
    coefficients = torch.linalg.solve(safe, target[triples])
    u, v = coefficients[:, 1], coefficients[:, 2]
    u_norm = torch.linalg.vector_norm(u, dim=-1)
    v_norm = torch.linalg.vector_norm(v, dim=-1)
    valid &= (u_norm > 1) & (v_norm > 1) & (
        torch.linalg.vector_norm(torch.cross(u, v, dim=-1), dim=-1)
        > 1e-3 * u_norm * v_norm)
    movement = torch.linalg.vector_norm(anchors @ (coefficients - old), dim=-1).mean(-1)
    valid &= movement <= 2500
    diagnostics['valid_triples'] = int(valid.sum())
    if not bool(valid.any()):
        return state, diagnostics

    residual = torch.linalg.vector_norm(design @ coefficients - target, dim=-1)
    score = ((1 - residual / 800).clamp_min(0) * weight).sum(-1) / total
    best = score.masked_fill(~valid, -torch.inf).argmax()
    diagnostics['consensus_score'] = float(score[best])
    diagnostics['five_anchor_movement_um'] = float(movement[best])
    if bool(score[best] < old_score + .05):
        return state, diagnostics

    inliers = (residual[best] < 800) & (weight > 0)
    diagnostics['inliers'] = int(inliers.sum())
    weighted = weight * inliers
    gram = design.T @ (weighted[:, None] * design)
    eigenvalues = torch.linalg.eigvalsh(gram)
    if bool(eigenvalues[0] <= 1e-6 * eigenvalues[-1]):
        return state, diagnostics
    refined = torch.linalg.solve(gram, design.T @ (weighted[:, None] * target))
    u, v = refined[1], refined[2]
    u_norm, v_norm = torch.linalg.vector_norm(u), torch.linalg.vector_norm(v)
    if bool((u_norm <= 1) | (v_norm <= 1) | (
        torch.linalg.vector_norm(torch.cross(u, v, dim=-1)) <= 1e-3 * u_norm * v_norm)):
        return state, diagnostics
    refined_movement = torch.linalg.vector_norm(anchors @ (refined - old), dim=-1).mean()
    final_residual = torch.linalg.vector_norm(design @ refined - target, dim=-1)
    final_score = ((1 - final_residual / 800).clamp_min(0) * weight).sum() / total
    if bool((refined_movement > 2500) | (final_score < old_score + .05)):
        return state, diagnostics

    origin = refined[0] - .5 * (refined[1] + refined[2])
    fitted = full_frame_state_from_components(*physical_ouv_to_frame(
        torch.stack((origin, refined[1], refined[2]))[None]))
    diagnostics.update(updated=True, accepted=True, final_score=float(final_score),
                       five_anchor_movement_um=float(refined_movement))
    return fitted, diagnostics
