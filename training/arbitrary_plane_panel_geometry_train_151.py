"""TRAIN-only v7 inverse-map sections with exact-v6 panel plane/framing draws.

Pass the result as ``source_sample`` to the frozen v3/v4 artifact functions.
Input indices address the 64 independent TRAIN deformation bases, not their
4,096 virtual affine descendants. The inverse maps remain approximate FP32
targets; the geometry/appearance distribution, not the map solver, changes.
No hidden tissue or eligibility retries occur; the caller records all draws.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_streaming_synthetic_v7 import MAP_ROOT, MODES
from training.arbitrary_plane_streaming_synthetic_v7_64 import NEW_ROOT
from training.arbitrary_plane_streaming_synthetic_v7_appearance_v3 import MODE_PROBS

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sample_panel_geometry_train_151(context, base_indices, seed, side=256):
    """Independent full-box planes/appearance on the 64 frozen TRAIN maps."""
    device = context['atlas'].device
    subjects = [context['bases'][int(index)] for index in base_indices]
    assert len(context['bases']) == 64 and all(s['lineage']['split'] == 'train' for s in subjects)
    planes, offsets, reflections, modes, appearances, records = [], [], [], [], [], []
    illuminations, textures, noises = [], [], []
    for row, (base_index, base) in enumerate(zip(base_indices, subjects)):
        base_index = int(base_index)
        prefix = [int(seed), base_index, row]
        rng = np.random.default_rng(np.random.SeedSequence([*prefix, 0]))
        appearance_rng = np.random.default_rng(np.random.SeedSequence([*prefix, 1]))
        noise_seed = int(np.random.SeedSequence([*prefix, 2]).generate_state(1, dtype=np.uint64)[0])
        noise_rng = torch.Generator().manual_seed(noise_seed)

        bounds = np.asarray(base['bounds'], dtype=np.float64)
        box_centre, half_extent = bounds.mean(0), np.diff(bounds, axis=0)[0] / 2
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        normal *= 1 if normal[np.abs(normal).argmax()] >= 0 else -1
        reference = np.eye(3)[np.abs(normal).argmin()]
        u = reference - np.dot(reference, normal) * normal
        u /= np.linalg.norm(u)
        roll = float(rng.uniform(-np.pi, np.pi))
        u = np.cos(roll) * u + np.sin(roll) * np.cross(normal, u)
        v = np.cross(normal, u)
        thickness = float(rng.uniform(25, 100))
        z = np.linspace(-.5, .5, 9) * thickness
        radius = np.abs(normal) @ half_extent
        signed_offset = float(rng.uniform(-radius - z.max(), radius - z.min()))
        edge_u = u * (2 * (np.abs(u) @ half_extent) * side / (side - 1))
        edge_v = v * (2 * (np.abs(v) @ half_extent) * side / (side - 1))
        origin = (box_centre + signed_offset * normal
                  - (side - 1) / (2 * side) * (edge_u + edge_v))
        reflection = bool(rng.integers(2))

        mode = MODES[int(appearance_rng.choice(3, p=MODE_PROBS))]
        p = {'gain': float(appearance_rng.uniform(.6, 1.4)),
             'gamma': float(np.exp(appearance_rng.uniform(np.log(.6), np.log(1.6)))),
             'invert': bool(appearance_rng.integers(2)),
             'noise_std': float(appearance_rng.uniform(.005, .05)),
             'background_mean': float(appearance_rng.uniform(0, .8)),
             'background_slope_yx': appearance_rng.uniform(-.15, .15, 2).tolist(),
             'illumination': float(appearance_rng.uniform(0, .15)),
             'texture': float(appearance_rng.uniform(0, .06)),
             'mask_dilate': bool(appearance_rng.integers(2)),
             'mask_radius': int(appearance_rng.integers(1, max(2, side // 32 + 1))),
             'damage': bool(appearance_rng.random() < .2),
             'ellipse': [*appearance_rng.uniform(-.7, .7, 2),
                         *appearance_rng.uniform(.05, .25, 2)]}
        map_root = MAP_ROOT if base_index < 8 else NEW_ROOT
        map_path = map_root / f'train/subject_{base_index:08d}/inverse_map.npz'
        planes.append(np.stack((origin, edge_u, edge_v)))
        offsets.append(z)
        reflections.append(reflection)
        modes.append(mode)
        appearances.append(p)
        illuminations.append(torch.randn(1, 5, 5, generator=noise_rng))
        textures.append(torch.randn(1, 16, 16, generator=noise_rng))
        noises.append(torch.randn(side, side, generator=noise_rng))
        records.append({'physical_section_id':
            f'panel-geometry-train-151-seed-{seed}-row-{row}-base-{base_index}',
            'split': 'train', 'base_index': base_index,
            'base_lineage': base['lineage'],
            'subject_plan_receipt_sha256': base['plan_receipt'],
            'inverse_map_sha256': context['bindings'][str(map_path)],
            'atlas_binding': context['provenance']['atlas_binding'],
            'sampler_code_sha256': CODE_SHA256,
            'seed_prefix': prefix, 'seed_branches': 'geometry0,appearance1,TorchCPU-noise2',
            'noise_seed_uint64': noise_seed,
            'physical_ouv_um': planes[-1].tolist(),
            'physical_unit_normal_ap_dv_ml': normal.tolist(),
            'plane_roll_rad': roll, 'signed_offset_from_box_centre_um': signed_offset,
            'thickness_um': thickness, 'horizontal_reflection': reflection,
            'sampling_branch': 'uniform_box_slab_offset',
            'framing': {'kind': 'conservative_subject_box'},
            'mode': mode, 'appearance': p,
            'coordinate_truth_kind': 'baked_inverse_map_approximate'})

    count = len(subjects)
    ouv = torch.as_tensor(np.stack(planes), device=device, dtype=torch.float32)
    z = torch.as_tensor(np.stack(offsets), device=device, dtype=torch.float32)
    weights = z.new_tensor([1, 2, 2, 2, 2, 2, 2, 2, 1])[None].expand(count, -1) / 16
    flags = torch.as_tensor(reflections, device=device, dtype=torch.bool)
    axis = torch.arange(side, device=device) / side
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    physical_centre = (ouv[:, None, None, 0] + xx[None, :, :, None] * ouv[:, None, None, 1]
                       + yy[None, :, :, None] * ouv[:, None, None, 2])
    normal = F.normalize(torch.cross(ouv[:, 1], ouv[:, 2], dim=-1), dim=-1)
    physical_slab = physical_centre[:, None] + z[:, :, None, None, None] * normal[:, None, None, None]

    canonical_slab = torch.empty_like(physical_slab)
    for base_index in sorted({int(index) for index in base_indices}):
        rows = [row for row, index in enumerate(base_indices) if int(index) == base_index]
        base = context['bases'][base_index]
        unscaled = (base['map_centre'] +
                    (physical_slab[rows] - base['map_centre']) / base['map_scale'])
        grid = ((unscaled - base['lower']) / (base['upper'] - base['lower']) * 2 - 1).flip(-1)
        displacement = base['displacement_cpu'].to(device)
        delta = F.grid_sample(displacement[None].expand(len(rows), -1, -1, -1, -1),
                              grid, mode='bilinear', padding_mode='zeros',
                              align_corners=True).permute(0, 2, 3, 4, 1)
        canonical_slab[rows] = unscaled + delta
        del displacement, delta

    centre = canonical_slab[:, 4]
    centred = axis - axis.mean()
    edge_u = (centre * centred[None, None, :, None]).sum((1, 2)) / (side * centred.square().sum())
    edge_v = (centre * centred[None, :, None, None]).sum((1, 2)) / (side * centred.square().sum())
    origin = centre.mean((1, 2)) - axis.mean() * (edge_u + edge_v)
    state = full_frame_state_from_components(*physical_ouv_to_frame(torch.stack((origin, edge_u, edge_v), 1)))
    observed_slab = torch.where(flags[:, None, None, None, None], canonical_slab.flip(-2), canonical_slab)
    rendered = render_finite_thickness_coordinate_grid(
        context['atlas'], observed_slab, (0., 0., 0.), (25., 25., 25.), weights)
    clean, support = rendered[:, 0], rendered[:, 1].clamp(0, 1)

    illumination = F.interpolate(torch.stack(illuminations).to(device), (side, side),
                                 mode='bilinear', align_corners=False)[:, 0]
    texture = F.interpolate(torch.stack(textures).to(device), (side, side),
                            mode='bilinear', align_corners=False)[:, 0]
    noise = torch.stack(noises).to(device)
    gain = clean.new_tensor([p['gain'] for p in appearances])[:, None, None]
    gamma = clean.new_tensor([p['gamma'] for p in appearances])[:, None, None]
    inversion = torch.as_tensor([p['invert'] for p in appearances], device=device)[:, None, None]
    tissue = (clean / support.clamp_min(1e-6)).clamp(0, 1).pow(gamma)
    tissue = torch.where(inversion, 1 - tissue, tissue)
    tissue = (tissue * gain * (1 + illumination * clean.new_tensor(
        [p['illumination'] for p in appearances])[:, None, None])).clamp(0, 1)
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, side, device=device),
                            torch.linspace(-1, 1, side, device=device), indexing='ij')
    ellipses = clean.new_tensor([p['ellipse'] for p in appearances])
    damage = torch.as_tensor([p['damage'] for p in appearances], device=device)
    hole = (((gy - ellipses[:, 0, None, None]) / ellipses[:, 2, None, None]).square() +
            ((gx - ellipses[:, 1, None, None]) / ellipses[:, 3, None, None]).square())
    retained = ~damage[:, None, None] | (hole >= 1)
    visible = support * retained
    slope = clean.new_tensor([p['background_slope_yx'] for p in appearances])
    background = (clean.new_tensor([p['background_mean'] for p in appearances])[:, None, None]
                  + slope[:, 0, None, None] * gy + slope[:, 1, None, None] * gx
                  + texture * clean.new_tensor([p['texture'] for p in appearances])[:, None, None])
    before = (tissue * visible + background * (1 - visible)
              + noise * clean.new_tensor([p['noise_std'] for p in appearances])[:, None, None]).clamp(0, 1)
    masks = (support > 0) & retained
    for row, (mode, p) in enumerate(zip(modes, appearances)):
        if mode == 'raw':
            masks[row] = True
        elif mode == 'imperfect_brush':
            radius = p['mask_radius']
            value = masks[row][None, None].float()
            masks[row] = (F.max_pool2d(value, 2 * radius + 1, 1, radius)[0, 0] > 0
                          if p['mask_dilate'] else
                          -F.max_pool2d(-F.pad(value, (radius,) * 4),
                                        2 * radius + 1, 1)[0, 0] > 0)
    eroded = masks.clone()
    eroded[:, 1:] &= masks[:, :-1]
    eroded[:, :-1] &= masks[:, 1:]
    eroded[:, :, 1:] &= masks[:, :, :-1]
    eroded[:, :, :-1] &= masks[:, :, 1:]
    eroded[:, [0, -1], :] = False
    eroded[:, :, [0, -1]] = False
    available = torch.as_tensor([mode != 'raw' for mode in modes], device=device)
    image = before * masks
    inputs = torch.stack((image, ((masks & ~eroded) & available[:, None, None]).float(),
                          available[:, None, None].expand(-1, side, side).float(),
                          torch.zeros_like(image), torch.zeros_like(image)), 1)
    visible = visible * masks
    mass = visible.sum((1, 2))
    eligible = mass >= side * side / 144
    for row, record in enumerate(records):
        record['eligible'] = bool(eligible[row])
        record['visible_support_mass'] = float(mass[row])
    return {'inputs': inputs, 'state': state, 'reflection': flags.long(),
            'offsets': z, 'weights': weights, 'centre': observed_slab[:, 4],
            'support': support, 'visible': visible, 'masks': masks,
            'eligible': eligible, 'provenance': records}
