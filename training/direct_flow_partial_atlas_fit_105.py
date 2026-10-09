"""Continuous local CCF flow from the 094 image/atlas correlation field."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.spatial_joint_fit_094 import SpatialJointFit094


class DirectFlowPartialAtlasFit105(SpatialJointFit094):
    def __init__(self, matcher):
        super().__init__(matcher)
        self.field_input = nn.Sequential(
            nn.Conv2d(225 * 2 + 64 + 3, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.field_gates = nn.Conv2d(128, 128, 3, padding=1)
        self.field_proposal = nn.Conv2d(128, 64, 3, padding=1)
        self.field_output = nn.Conv2d(64, 230, 1)
        nn.init.zeros_(self.field_output.weight)
        nn.init.zeros_(self.field_output.bias)

    def forward(self, feature, state, reflection, atlas, offsets, weights,
                source_shape=(256, 256)):
        base = super().forward(feature, state, reflection, atlas, offsets, weights,
                               source_shape=source_shape)
        batch, count, _, side, _ = base['fine_match_logits'].shape
        logits = base['fine_match_logits'].flatten(0, 1)
        support = base['fine_match_support'].flatten(0, 1)
        image = F.adaptive_avg_pool2d(feature, (side, side))[:, None].expand(
            -1, count, -1, -1, -1).flatten(0, 1)
        validity = base['validity_logit'][:, None].expand(-1, count, -1, -1)
        y, x = torch.meshgrid((torch.arange(side, device=feature.device,
            dtype=feature.dtype) + .5) / side, (torch.arange(side,
            device=feature.device, dtype=feature.dtype) + .5) / side, indexing='ij')
        position = torch.stack((x - .5, y - .5))[None].expand(
            batch * count, -1, -1, -1)
        field = self.field_input(torch.cat((logits.clamp(-40, 40) / 20, support,
            image, validity.flatten(0, 1)[:, None], position), 1))
        hidden = torch.zeros_like(field)
        for _ in range(3):
            update, reset = self.field_gates(torch.cat((field, hidden), 1)).sigmoid().chunk(2, 1)
            proposed = self.field_proposal(torch.cat((field, reset * hidden), 1)).tanh()
            hidden = (1 - update) * hidden + update * proposed
        residual = self.field_output(hidden)
        match_logits = (logits + residual[:, :225]).masked_fill(support < .5, -1e4)
        supported = (support >= .5).sum(1).clamp_min(1)
        match_logits = match_logits - supported.log()[:, None]
        complete_logits = torch.cat((match_logits, residual[:, 225:226]), 1)
        unmatched = complete_logits.softmax(1)[:, 225]
        identity = residual[:, 229].sigmoid()
        mx = 2 * residual[:, 226].tanh()
        my = 2 * residual[:, 227].tanh()
        mz = 4 * residual[:, 228].tanh()
        fitted_state = base['state'].flatten(0, 1)
        centre, frame, basis = full_frame_state_to_components(fitted_state)
        edges = frame[:, :, :2] @ basis
        reflected = reflection.flatten().bool()[:, None, None]
        chart = torch.stack((torch.where(reflected,
            (source_shape[1] - 1) / source_shape[1] - (x - .5 / source_shape[1]),
            x - .5 / source_shape[1]),
            (y - .5 / source_shape[0]).expand(batch * count, -1, -1)), -1)
        shift = torch.stack((torch.where(reflected, -mx, mx) / side, my / side), -1)
        rigid = centre[:, None, None] + torch.einsum(
            'nij,nhwj->nhwi', edges, chart - .5)
        mapped = (rigid + torch.einsum('nij,nhwj->nhwi', edges, shift)
            + 1500 * mz[..., None] * frame[:, None, None, :, 2])
        heldout_axis = torch.arange(side, device=feature.device)
        iy, ix = torch.meshgrid(heldout_axis, heldout_axis, indexing='ij')
        heldout = ((iy // 2 + ix // 2) % 2 == 1).to(feature.dtype)
        tissue = validity.flatten(0, 1).sigmoid()
        fit_weight = tissue * (1 - unmatched) * (1 - heldout)
        design = torch.cat((torch.ones_like(chart[..., :1]), chart - .5), -1)
        ridge = torch.diag(feature.new_tensor((8., 1., 1.)))
        lhs = torch.einsum('nhwi,nhw,nhwj->nij', design, fit_weight, design) + ridge
        rhs = torch.einsum('nhwi,nhw,nhwj->nij', design, fit_weight, mapped - rigid)
        correction = torch.linalg.solve(lhs, rhs)
        centre_delta = correction[:, 0]
        centre_delta = centre_delta * (2500 / (2500 + centre_delta.norm(dim=-1, keepdim=True)))
        edge_delta = correction[:, 1:].transpose(1, 2)
        edge_limit = .1 * edges.norm(dim=-2, keepdim=True)
        edge_delta = edge_delta * edge_limit / (edge_limit + edge_delta.norm(dim=-2, keepdim=True))
        fit_centre = centre + centre_delta
        fit_edges = edges + edge_delta
        u, v = fit_edges.unbind(-1)
        coherent_state = full_frame_state_from_components(*physical_ouv_to_frame(
            torch.stack((fit_centre - .5 * (u + v), u, v), -2)))
        true_centre, true_frame, true_basis = full_frame_state_to_components(coherent_state)
        true_edges = true_frame[:, :, :2] @ true_basis
        local = mapped - (true_centre[:, None, None] + torch.einsum(
            'nij,nhwj->nhwi', true_edges, chart - .5))
        heldout_weight = tissue * heldout
        quality = ((1 - unmatched) * heldout_weight).sum((-2, -1)) / \
            heldout_weight.sum((-2, -1)).clamp_min(1e-6)
        local_mm = local / 1000
        deformation_cost = (local_mm.square().sum(-1) * heldout_weight).sum((-2, -1)) / \
            heldout_weight.sum((-2, -1)).clamp_min(1e-6)
        scale_x, scale_y = (true_edges.norm(dim=1) / side / 1000).clamp_min(.1).unbind(-1)
        strain_x = (local_mm[:, :, 1:] - local_mm[:, :, :-1]).square().sum(-1) / scale_x[:, None, None].square()
        strain_y = (local_mm[:, 1:] - local_mm[:, :-1]).square().sum(-1) / scale_y[:, None, None].square()
        x_weight = heldout_weight[:, :, 1:] * heldout_weight[:, :, :-1]
        y_weight = heldout_weight[:, 1:] * heldout_weight[:, :-1]
        strain_cost = .5 * ((strain_x * x_weight).sum((-2, -1)) / x_weight.sum((-2, -1)).clamp_min(1e-6)
            + (strain_y * y_weight).sum((-2, -1)) / y_weight.sum((-2, -1)).clamp_min(1e-6))
        global_cost = (centre_delta / 2500).square().sum(-1) + \
            (edge_delta / edge_limit.clamp_min(1)).square().mean((1, 2))
        energy = -quality.clamp_min(1e-4).log() + .1 * deformation_cost + \
            .02 * strain_cost + .02 * global_cost
        return {**base, 'coherent_match_logits': complete_logits.reshape(
            batch, count, 226, side, side),
            'coherent_unmatched_probability': unmatched.reshape(batch, count, side, side),
            'coherent_identity_probability': identity.reshape(batch, count, side, side),
            'coherent_offset_cells_xyz': torch.stack((mx, my, mz), 1).reshape(
                batch, count, 3, side, side),
            'coherent_state': coherent_state.reshape(batch, count, 12),
            'coherent_local_displacement_um': local.reshape(batch, count, side, side, 3),
            'coherent_match_ccf_um': mapped.reshape(batch, count, side, side, 3),
            'coherent_quality': quality.reshape(batch, count),
            'coherent_deformation_cost': deformation_cost.reshape(batch, count),
            'coherent_strain_cost': strain_cost.reshape(batch, count),
            'coherent_global_cost': global_cost.reshape(batch, count),
            'coherent_energy': energy.reshape(batch, count)}
