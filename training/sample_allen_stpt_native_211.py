"""Independent arbitrary-plane virtual cuts through one acquired TRAIN STPT specimen."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_streaming_synthetic_v7 import MODES


def load_allen_stpt_native_211(donor_dir, map_path, device='cuda', inner_role='train'):
    donor_dir, map_path = Path(donor_dir), Path(map_path)
    geometry = json.loads((donor_dir / 'native_stack_geometry.json').read_text())
    stack = np.load(donor_dir / 'native_red_50um_xy_100um_z.npy', mmap_mode='r')
    mapped = np.load(map_path, mmap_mode='r')
    assert mapped.shape == (*stack.shape, 3)
    assert geometry['axis_order'] == 'zyx' and geometry['section_thickness_um'] == 100
    spacing_x, spacing_y, spacing_z = geometry['native_spacing_um_xyz']
    first = geometry['section_numbers'][0]
    affine = np.asarray(geometry['alignment3d_tvr'], dtype=np.float32)
    linear, translation = affine[:9].reshape(3, 3), affine[9:]
    x = ((np.arange(stack.shape[2], dtype=np.float32) + .5) * spacing_x)[None, None, :]
    y = ((np.arange(stack.shape[1], dtype=np.float32) + .5) * spacing_y)[None, :, None]
    z = ((np.arange(stack.shape[0], dtype=np.float32) + first) * spacing_z)[:, None, None]
    residual = np.empty(mapped.shape, dtype=np.float32)
    for axis in range(3):
        residual[..., axis] = mapped[..., axis] - (linear[axis, 0] * x +
            linear[axis, 1] * y + linear[axis, 2] * z + translation[axis])
    occupancy = np.empty(stack.shape, dtype=np.uint8)
    for section in range(stack.shape[0]):
        components, _ = label(stack[section] > 5)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        occupancy[section] = (sizes >= 25)[components]
    points = np.argwhere(occupancy[::3, ::4, ::4] > 0).astype(np.float32)
    points *= (3, 4, 4)
    points = np.column_stack(((points[:, 2] + .5) * spacing_x,
                              (points[:, 1] + .5) * spacing_y,
                              (points[:, 0] + first) * spacing_z))
    extent = np.array([stack.shape[2] * spacing_x, stack.shape[1] * spacing_y,
                       stack.shape[0] * spacing_z], dtype=np.float32)
    lower = np.array([0., 0., (first - .5) * spacing_z], dtype=np.float32)
    upper = lower + extent
    volume = torch.from_numpy(np.array(stack, copy=True)).to(device, torch.float32)[None, None] / 255
    return {'volume': volume,
            'occupancy': torch.from_numpy(occupancy)[None, None].to(device, torch.float32),
            'residual': torch.from_numpy(residual).permute(3, 0, 1, 2)[None].to(device),
            'affine': torch.from_numpy(linear).to(device),
            'translation': torch.from_numpy(translation).to(device),
            'spacing': (spacing_x, spacing_y, spacing_z), 'first': first,
            'bounds': (lower, upper), 'support_points': points,
            'geometry': geometry, 'map_path': str(map_path), 'inner_role': inner_role}


@torch.no_grad()
def sample_allen_stpt_native_211(context, donor_indices, seed, side=256):
    count, device = len(donor_indices), context['volume'].device
    assert all(int(i) == context['geometry']['animal_id'] for i in donor_indices)
    lower, upper = context['bounds']
    box_centre, half_extent = (lower + upper) / 2, (upper - lower) / 2
    rows, offsets, reflections, parameters, records = [], [], [], [], []
    for row, donor_id in enumerate(donor_indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(donor_id), row, 0]))
        appearance = np.random.default_rng(np.random.SeedSequence([int(seed), int(donor_id), row, 1]))
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        normal *= 1 if normal[np.abs(normal).argmax()] >= 0 else -1
        u = np.eye(3)[np.abs(normal).argmin()]
        u -= np.dot(u, normal) * normal
        u /= np.linalg.norm(u)
        roll = rng.uniform(-np.pi, np.pi)
        u = np.cos(roll) * u + np.sin(roll) * np.cross(normal, u)
        v = np.cross(normal, u)
        thickness = float(rng.uniform(25, 100))
        axial = np.linspace(-.5, .5, 9) * thickness
        if rng.random() < .8:
            anchor = context['support_points'][rng.integers(len(context['support_points']))]
            distance = np.dot(normal, anchor - box_centre)
            sampling = 'through_acquired_tissue_voxel'
        else:
            distance = rng.uniform(-np.abs(normal) @ half_extent,
                                    np.abs(normal) @ half_extent)
            sampling = 'uniform_specimen_box_offset'
        centre = box_centre + distance * normal + rng.normal(0, 250, 1) * u + rng.normal(0, 250, 1) * v
        edge_u = 2 * (np.abs(u) @ half_extent) * side / (side - 1) * np.exp(rng.normal(0, .12)) * u
        edge_v = 2 * (np.abs(v) @ half_extent) * side / (side - 1) * np.exp(rng.normal(0, .12)) * (v + rng.normal(0, .03) * u)
        origin = centre - (side - 1) / (2 * side) * (edge_u + edge_v)
        reflected = bool(rng.integers(2))
        mode = int(appearance.choice(3, p=(.7, .15, .15)))
        p = {'gain': float(appearance.uniform(.7, 1.3)),
             'gamma': float(appearance.uniform(.8, 1.2)),
             'invert': bool(appearance.random() < .25),
             'noise_std': float(appearance.uniform(0, .02)),
             'background_mean': float(appearance.uniform(0, .6)),
             'background_slope_yx': appearance.uniform(-.12, .12, 2).tolist(),
             'mask_radius': int(appearance.integers(1, 5)),
             'mask_dilate': bool(appearance.integers(2))}
        rows.append(np.stack((origin, edge_u, edge_v)))
        offsets.append(axial)
        reflections.append(reflected)
        parameters.append((mode, p))
        records.append({'physical_section_id': f"native-virtual-{donor_id}-{seed}-{row}",
                        'virtual_plane_id': f"native-virtual-{donor_id}-{seed}-{row}",
                        'animal_id': int(donor_id),
                        'specimen_id': context['geometry']['specimen_id'],
                        'experiment_id': context['geometry']['experiment_id'],
                        'source_section_numbers': [context['geometry']['section_numbers'][0],
                                                   context['geometry']['section_numbers'][-1]],
                        'reserved_split': 'train', 'inner_role': context['inner_role'],
                        'shape_h_w': [side, side],
                        'seed': int(seed), 'row': row, 'sampling_branch': sampling,
                        'native_ouv_um': rows[-1].tolist(), 'native_unit_normal': normal.tolist(),
                        'virtual_thickness_um': thickness, 'horizontal_reflection': reflected,
                        'mode': MODES[mode], 'appearance': p,
                        'coordinate_truth_kind': 'Allen_authored_weak_native_to_CCF_map_interpolated',
                        'oblique_evidence_kind': 'virtual_interpolation_of_100um_coronal_sections',
                        'tissue_support_kind': 'native_red_gt5_connected_components_at_least25_pixels_per_acquired_section',
                        'native_map_path': context['map_path']})
    ouv = torch.as_tensor(np.stack(rows), device=device, dtype=torch.float32)
    axial = torch.as_tensor(np.stack(offsets), device=device, dtype=torch.float32)
    weights = axial.new_tensor([1, 2, 2, 2, 2, 2, 2, 2, 1])[None].expand(count, -1) / 16
    reflection = torch.as_tensor(reflections, device=device, dtype=torch.bool)
    yy, xx = torch.meshgrid(torch.arange(side, device=device) / side,
                            torch.arange(side, device=device) / side, indexing='ij')
    native_centre = ouv[:, None, None, 0] + xx[None, :, :, None] * ouv[:, None, None, 1] + yy[None, :, :, None] * ouv[:, None, None, 2]
    normal = F.normalize(torch.cross(ouv[:, 1], ouv[:, 2], dim=-1), dim=-1)
    native_slab = native_centre[:, None] + axial[:, :, None, None, None] * normal[:, None, None, None]
    sx, sy, sz = context['spacing']
    depth, height, width = context['volume'].shape[-3:]
    index = torch.stack((native_slab[..., 0] / sx - .5,
                         native_slab[..., 1] / sy - .5,
                         native_slab[..., 2] / sz - context['first']), -1)
    grid = torch.stack((2 * index[..., 0] / (width - 1) - 1,
                        2 * index[..., 1] / (height - 1) - 1,
                        2 * index[..., 2] / (depth - 1) - 1), -1)
    source = F.grid_sample(context['volume'].expand(count, -1, -1, -1, -1),
                           grid, mode='bilinear', padding_mode='zeros', align_corners=True)[:, 0]
    support = F.grid_sample(context['occupancy'].expand(count, -1, -1, -1, -1),
                            grid, mode='bilinear', padding_mode='zeros', align_corners=True)[:, 0]
    residual = F.grid_sample(context['residual'].expand(count, -1, -1, -1, -1),
                             grid, mode='bilinear', padding_mode='border', align_corners=True)
    canonical_slab = native_slab @ context['affine'].T + context['translation'] + residual.permute(0, 2, 3, 4, 1)
    canonical_centre = canonical_slab[:, 4]
    st = torch.arange(side, device=device) / side
    centred = st - st.mean()
    edge_u = (canonical_centre * centred[None, None, :, None]).sum((1, 2)) / (side * centred.square().sum())
    edge_v = (canonical_centre * centred[None, :, None, None]).sum((1, 2)) / (side * centred.square().sum())
    origin = canonical_centre.mean((1, 2)) - st.mean() * (edge_u + edge_v)
    state = full_frame_state_from_components(*physical_ouv_to_frame(torch.stack((origin, edge_u, edge_v), 1)))
    observed_centre = torch.where(reflection[:, None, None, None], canonical_centre.flip(-2), canonical_centre)
    source = torch.where(reflection[:, None, None, None], source.flip(-1), source)
    support = torch.where(reflection[:, None, None, None], support.flip(-1), support)
    clean = (source * weights[:, :, None, None]).sum(1)
    support = (support * weights[:, :, None, None]).sum(1).clamp(0, 1)
    mode = torch.tensor([p[0] for p in parameters], device=device)
    gain = torch.tensor([p[1]['gain'] for p in parameters], device=device)[:, None, None]
    gamma = torch.tensor([p[1]['gamma'] for p in parameters], device=device)[:, None, None]
    inverted = torch.tensor([p[1]['invert'] for p in parameters], device=device)[:, None, None]
    tissue = (clean / support.clamp_min(.05)).clamp(0, 1)
    tissue = torch.where(inverted, 1 - tissue, tissue).pow(gamma).mul(gain).clamp(0, 1)
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, side, device=device),
                            torch.linspace(-1, 1, side, device=device), indexing='ij')
    slope = torch.tensor([p[1]['background_slope_yx'] for p in parameters], device=device)
    mean = torch.tensor([p[1]['background_mean'] for p in parameters], device=device)
    background = (mean[:, None, None] + slope[:, 0, None, None] * gy + slope[:, 1, None, None] * gx).clamp(0, 1)
    noise = torch.stack([torch.randn(side, side, generator=torch.Generator().manual_seed(
        int(np.random.SeedSequence([int(seed), int(donor_id), row, 2]).generate_state(1)[0])))
        for row, donor_id in enumerate(donor_indices)]).to(device)
    image = (tissue * support + (background + .25 * clean) * (1 - support) +
             noise * torch.tensor([p[1]['noise_std'] for p in parameters], device=device)[:, None, None]).clamp(0, 1)
    masks = support > .5
    for row, (kind, p) in enumerate(parameters):
        if kind == 0:
            masks[row] = True
        elif kind == 2:
            radius = p['mask_radius']
            source_mask = masks[row][None, None].float()
            masks[row] = (F.max_pool2d(source_mask, 2 * radius + 1, 1, radius)[0, 0] > 0 if p['mask_dilate']
                          else 1 - F.max_pool2d(1 - source_mask, 2 * radius + 1, 1, radius)[0, 0] > .5)
    available = mode != 0
    eroded = masks.clone()
    eroded[:, 1:] &= masks[:, :-1]
    eroded[:, :-1] &= masks[:, 1:]
    eroded[:, :, 1:] &= masks[:, :, :-1]
    eroded[:, :, :-1] &= masks[:, :, 1:]
    eroded[:, [0, -1], :] = False
    eroded[:, :, [0, -1]] = False
    outline = (masks & ~eroded) & available[:, None, None]
    image = image * masks
    inputs = torch.stack((image, outline.float(), available[:, None, None].expand(-1, side, side).float(),
                          torch.zeros_like(image), torch.zeros_like(image)), 1)
    visible = support * masks
    eligible = visible.sum((1, 2)) >= side * side / 144
    for row, record in enumerate(records):
        record['eligible'] = bool(eligible[row])
        record['visible_support_mass'] = float(visible[row].sum())
    return {'inputs': inputs, 'state': state, 'reflection': reflection.long(),
            'offsets': axial, 'weights': weights, 'centre': observed_centre,
            'support': support, 'visible': visible, 'masks': masks,
            'eligible': eligible, 'provenance': records}
