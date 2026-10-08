"""Frozen v3 appearance draw on the v7 arbitrary-plane geometry; old v7 stays immutable."""

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, render_finite_thickness_coordinate_grid
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_streaming_synthetic_v7 import MODES

MODE_PROBS = (.7, .15, .15)


def sample_streaming_synthetic_v7_appearance_v3(context, subject_indices, seed, side=192):
    """Generate fresh physical sections from512 virtual subject indices.

    No output files or hidden retries. `seed`, row order, virtual IDs, side and
    bound context replay the complete batch. Caller excludes/logs ineligible
    rows, and continues drawing until its unique-informative-plane budget is met.
    """
    subjects = [context['subjects'][int(i)] for i in subject_indices]
    count, device = len(subjects), context['atlas'].device
    ouv_rows, offsets, reflections, records, parameters, modes, noises, illuminations, textures = [], [], [], [], [], [], [], [], []
    for row, subject in enumerate(subjects):
        prefix = [int(seed), subject['virtual_index'], row]
        rng = np.random.default_rng(np.random.SeedSequence([*prefix, 0]))
        appearance = np.random.default_rng(np.random.SeedSequence([*prefix, 1]))
        noise_seed = int(np.random.SeedSequence([*prefix, 2]).generate_state(1, dtype=np.uint64)[0])
        noise_rng = torch.Generator().manual_seed(noise_seed)
        base = context['bases'][subject['base_index']]
        bounds = np.array(subject['bounds_um'])
        box_centre, half_extent = bounds.mean(0), (bounds[1] - bounds[0]) / 2
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
        z = np.linspace(-.5, .5, 9) * thickness
        radius = np.abs(normal) @ half_extent
        support_index = None
        if rng.random() < .75:
            support_index = int(rng.integers(len(base['support_points'])))
            anchor, scale = np.array(subject['anchor_um']), np.array(subject['scale_ap_dv_ml'])
            support_point = anchor + scale * (base['support_points'][support_index] - anchor)
            offset = np.dot(normal, support_point - box_centre)
        else:
            offset = rng.uniform(-radius - z.max(), radius - z.min())
        centre = box_centre + offset * normal
        edge_u = u * (2 * (np.abs(u) @ half_extent) * side / (side - 1))
        edge_v = v * (2 * (np.abs(v) @ half_extent) * side / (side - 1))
        origin = centre - (side - 1) / (2 * side) * (edge_u + edge_v)
        framing = {'kind': 'conservative_virtual_subject_box'}
        if rng.random() < .5:
            donor_rows = context['donor_rows'][int(rng.integers(len(context['donor_rows'])))]
            geometry_row = int(rng.choice(donor_rows))
            shift_u, shift_v, lu, lv, shear = context['nuisance'][geometry_row].copy()
            shift_u, shift_v = np.array([shift_u, shift_v]) + rng.normal(0, 300, 2)
            lu, lv = np.array([lu, lv]) * np.exp(rng.normal(0, .12, 2))
            shear += rng.normal(0, .02)
            anchor = np.array(subject['anchor_um'])
            centre = anchor + normal * np.dot(normal, origin - anchor) + shift_u * u + shift_v * v
            edge_u, edge_v = lu * u, lv * (v + shear * u)
            origin = centre - .5 * (edge_u + edge_v)
            framing = {'kind': 'TRAIN_acquisition_tuple', 'source_geometry_row': geometry_row,
                'source_identity': {key: context['geometry'][geometry_row][key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                'shift_u_v_length_u_v_shear': [shift_u, shift_v, lu, lv, shear]}
        reflection = bool(rng.integers(2))
        mode = int(appearance.choice(3, p=MODE_PROBS))
        p = {'gain': float(appearance.uniform(.6, 1.4)), 'gamma': float(np.exp(appearance.uniform(np.log(.6), np.log(1.6)))),
            'invert': bool(appearance.integers(2)), 'noise_std': float(appearance.uniform(.005, .05)),
            'background_mean': float(appearance.uniform(0, .8)), 'background_slope_yx': appearance.uniform(-.15, .15, 2).tolist(),
            'illumination': float(appearance.uniform(0, .15)), 'texture': float(appearance.uniform(0, .06)),
            'mask_dilate': bool(appearance.integers(2)), 'mask_radius': int(appearance.integers(1, max(2, side // 32 + 1))),
            'damage': bool(appearance.random() < .2), 'ellipse': [*appearance.uniform(-.7, .7, 2), *appearance.uniform(.05, .25, 2)]}
        ouv_rows.append(np.stack((origin, edge_u, edge_v)))
        offsets.append(z)
        reflections.append(reflection)
        parameters.append(p)
        modes.append(mode)
        illuminations.append(torch.randn(1, 5, 5, generator=noise_rng))
        textures.append(torch.randn(1, 16, 16, generator=noise_rng))
        noises.append(torch.randn(side, side, generator=noise_rng))
        records.append({'physical_section_id': f"stream-v7-seed-{seed}-row-{row}-virtual-{subject['virtual_index']}",
            'seed_prefix': prefix, 'seed_branches': 'geometry0,appearance1,TorchCPU-noise2', 'noise_seed_uint64': noise_seed,
            'virtual_index': subject['virtual_index'], 'virtual_subject_id': subject['virtual_subject_id'],
            'base_lineage': subject['base_lineage'], 'split': 'train', 'shape_h_w': [side, side],
            'support_point_index': support_index, 'sampling_branch': 'through_support_point' if support_index is not None else 'uniform_box_offset',
            'virtual_ouv_um': ouv_rows[-1].tolist(), 'virtual_unit_normal': normal.tolist(), 'thickness_um': thickness,
            'horizontal_reflection': reflection, 'mode': MODES[mode], 'framing': framing, 'appearance': p,
            'coordinate_truth_kind': 'baked_inverse_map_approximate'})
    ouv = torch.tensor(np.array(ouv_rows), device=device, dtype=torch.float32)
    z = torch.tensor(np.array(offsets), device=device, dtype=torch.float32)
    weights = z.new_tensor([1, 2, 2, 2, 2, 2, 2, 2, 1])[None].expand(count, -1) / 16
    flags = torch.tensor(reflections, device=device, dtype=torch.bool)
    yy, xx = torch.meshgrid(torch.arange(side, device=device) / side, torch.arange(side, device=device) / side, indexing='ij')
    virtual_centre = ouv[:, None, None, 0] + xx[None, :, :, None] * ouv[:, None, None, 1] + yy[None, :, :, None] * ouv[:, None, None, 2]
    normal = F.normalize(torch.cross(ouv[:, 1], ouv[:, 2], dim=-1), dim=-1)
    virtual_slab = virtual_centre[:, None] + z[:, :, None, None, None] * normal[:, None, None, None]
    anchors = torch.tensor([s['anchor_um'] for s in subjects], device=device, dtype=torch.float32)
    scales = torch.tensor([s['scale_ap_dv_ml'] for s in subjects], device=device, dtype=torch.float32)
    base_queries = anchors[:, None, None, None] + (virtual_slab - anchors[:, None, None, None]) / scales[:, None, None, None]
    canonical_slab = torch.empty_like(base_queries)
    for base_index in sorted({s['base_index'] for s in subjects}):
        rows = [i for i, s in enumerate(subjects) if s['base_index'] == base_index]
        base = context['bases'][base_index]
        unscaled = base['map_centre'] + (base_queries[rows] - base['map_centre']) / base['map_scale']
        grid = ((unscaled - base['lower']) / (base['upper'] - base['lower']) * 2 - 1).flip(-1)
        delta = F.grid_sample(base['displacement'][None].expand(len(rows), -1, -1, -1, -1), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True).permute(0, 2, 3, 4, 1)
        canonical_slab[rows] = unscaled + delta
    centre = canonical_slab[:, 4]
    st = torch.arange(side, device=device) / side
    centred = st - st.mean()
    edge_u = (centre * centred[None, None, :, None]).sum((1, 2)) / (side * centred.square().sum())
    edge_v = (centre * centred[None, :, None, None]).sum((1, 2)) / (side * centred.square().sum())
    origin = centre.mean((1, 2)) - st.mean() * (edge_u + edge_v)
    canonical_ouv = torch.stack((origin, edge_u, edge_v), 1)
    components = physical_ouv_to_frame(canonical_ouv)
    state = full_frame_state_from_components(*components)
    frame = components[1]
    fitted_plane = origin[:, None, None] + xx[None, :, :, None] * edge_u[:, None, None] + yy[None, :, :, None] * edge_v[:, None, None]
    director = torch.einsum('bs,bshwc->bhwc', weights * z, canonical_slab - centre[:, None]) / (weights * z.square()).sum(1)[:, None, None, None]
    residual_local = torch.einsum('bji,bhwj->bihw', frame, centre - fitted_plane) / 200
    director_local = torch.einsum('bji,bhwj->bihw', frame, director - frame[:, None, None, :, 2]) / .1
    field = F.interpolate(torch.cat((residual_local, director_local), 1), (8, 8), mode='bilinear', align_corners=False)
    observed_slab = torch.where(flags[:, None, None, None, None], canonical_slab.flip(-2), canonical_slab)
    rendered = render_finite_thickness_coordinate_grid(context['atlas'], observed_slab, (0., 0., 0.), (25., 25., 25.), weights)
    clean, support = rendered[:, 0], rendered[:, 1].clamp(0, 1)
    illumination = F.interpolate(torch.stack(illuminations).to(device), (side, side), mode='bilinear', align_corners=False)[:, 0]
    texture = F.interpolate(torch.stack(textures).to(device), (side, side), mode='bilinear', align_corners=False)[:, 0]
    noise = torch.stack(noises).to(device)
    gain = clean.new_tensor([p['gain'] for p in parameters])[:, None, None]
    gamma = clean.new_tensor([p['gamma'] for p in parameters])[:, None, None]
    inversion = torch.tensor([p['invert'] for p in parameters], device=device)[:, None, None]
    appearance = (clean / support.clamp_min(1e-6)).clamp(0, 1).pow(gamma)
    appearance = torch.where(inversion, 1 - appearance, appearance)
    appearance = (appearance * gain * (1 + illumination * clean.new_tensor([p['illumination'] for p in parameters])[:, None, None])).clamp(0, 1)
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, side, device=device), torch.linspace(-1, 1, side, device=device), indexing='ij')
    ellipses = clean.new_tensor([p['ellipse'] for p in parameters])
    damage = torch.tensor([p['damage'] for p in parameters], device=device)
    retained = ~damage[:, None, None] | (((gy - ellipses[:, 0, None, None]) / ellipses[:, 2, None, None]).square()
        + ((gx - ellipses[:, 1, None, None]) / ellipses[:, 3, None, None]).square() >= 1)
    visible = support * retained
    slope = clean.new_tensor([p['background_slope_yx'] for p in parameters])
    background = clean.new_tensor([p['background_mean'] for p in parameters])[:, None, None] + slope[:, 0, None, None] * gy + slope[:, 1, None, None] * gx
    background = background + texture * clean.new_tensor([p['texture'] for p in parameters])[:, None, None]
    before_brush = (appearance * visible + background * (1 - visible)
        + noise * clean.new_tensor([p['noise_std'] for p in parameters])[:, None, None]).clamp(0, 1)
    masks = (support > 0) & retained
    for row, (mode, p) in enumerate(zip(modes, parameters)):
        if mode == 0:
            masks[row] = True
        elif mode == 2:
            radius = p['mask_radius']
            source = masks[row][None, None].float()
            masks[row] = (F.max_pool2d(source, 2 * radius + 1, 1, radius)[0, 0] > 0 if p['mask_dilate'] else
                -F.max_pool2d(-F.pad(source, (radius,) * 4), 2 * radius + 1, 1)[0, 0] > 0)
    eroded = masks.clone()
    eroded[:, 1:] &= masks[:, :-1]
    eroded[:, :-1] &= masks[:, 1:]
    eroded[:, :, 1:] &= masks[:, :, :-1]
    eroded[:, :, :-1] &= masks[:, :, 1:]
    eroded[:, [0, -1], :] = False
    eroded[:, :, [0, -1]] = False
    available = torch.tensor(modes, device=device) != 0
    outline = (masks & ~eroded) & available[:, None, None]
    image = before_brush * masks
    inputs = torch.stack((image, outline.float(), available[:, None, None].expand(-1, side, side).float(),
                          torch.zeros_like(image), torch.zeros_like(image)), 1)
    visible = visible * masks
    mass = visible.sum((1, 2))
    eligible = mass >= side * side / 144
    pixel = torch.unique(torch.cat((torch.arange(0, side, 4, device=device), torch.tensor([side - 1], device=device))))
    eligibility, masses = eligible.cpu().tolist(), mass.cpu().tolist()
    director_delta = director_local * .1
    capacity = torch.stack((
        (residual_local * 200).abs().flatten(1).amax(1),
        director_delta.abs().flatten(1).amax(1),
        (director_delta.abs() > .2).float().flatten(1).mean(1),
        (director_delta - director_delta.clamp(-.2, .2)).norm(dim=1).flatten(1).amax(1) * z.abs().amax(1),
    ), -1).cpu().tolist()
    for row, record in enumerate(records):
        record['eligible'], record['visible_support_mass'] = eligibility[row], masses[row]
        record['target_capacity'] = dict(zip(('residual_component_absmax_um', 'director_component_absmax',
            'director_components_outside_model_cap_fraction', 'director_cap_only_max_slab_displacement_um'), capacity[row]))
    return {'inputs': inputs, 'state': state, 'reflection': flags.long(), 'offsets': z, 'weights': weights,
        'clean': clean, 'support': support, 'retained': retained, 'masks': masks, 'visible': visible, 'field': field,
        'centre': observed_slab[:, 4], 'slab': observed_slab[:, :, pixel[:, None], pixel[None, :], :],
        'psf_pixel_y': pixel, 'psf_pixel_x': pixel, 'eligible': eligible, 'provenance': records}


def sample_streaming_synthetic_v7_64_appearance_v3(context, subject_indices, seed, side=192):
    selected = sorted({context["subjects"][int(i)]["base_index"] for i in subject_indices})
    device = context["atlas"].device
    for index in selected:
        base = context["bases"][index]
        base["displacement"] = base["displacement_cpu"].to(device)
    batch = sample_streaming_synthetic_v7_appearance_v3(context, subject_indices, seed, side)
    for index in selected:
        context["bases"][index].pop("displacement")
    return batch
