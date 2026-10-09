"""Virtual full-angle TRAIN-donor cuts with physical-pose v3 artifact labels.

These are reslices of acquired serial sections, not acquired oblique slides;
their atlas coordinates inherit weak Allen affines.
"""
import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import (
    sample_one_shot_slide_artifacts_v3,
)
from training.arbitrary_plane_streaming_synthetic_v7_appearance_v3 import MODE_PROBS


def sample_reserved_real_oblique_v3(context, donor_index, seed, side=192,
                                   device='cuda', return_source=False):
    donor_index = int(donor_index)
    donor = context['donors'][donor_index]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), donor_index, 37]))
    order = np.argsort(donor['state'][:, 0])
    states = torch.from_numpy(donor['state'][order].copy())
    center, frame, basis = full_frame_state_to_components(states)
    center, frame, basis = center.numpy(), frame.numpy(), basis.numpy()
    edge = frame[0, :, :2] @ basis[0]
    step = (center[-1] - center[0]) / (len(order) - 1)
    assert np.max(np.linalg.norm(center - center[0] - np.arange(len(order))[:, None] * step, axis=1)) < 5
    assert np.max(np.abs(states.numpy()[:, 3:] - states.numpy()[0, 3:])) < 1e-5
    inverse = np.linalg.inv(np.column_stack((edge[:, 0], edge[:, 1], step)))

    cache = context.setdefault('oblique_volume_cache', {})
    if donor_index not in cache:
        images = np.load(donor['images'], mmap_mode='r')
        cache[donor_index] = torch.from_numpy(
            images[order, :1].astype(np.float32).transpose(1, 0, 2, 3))[None]
    volume_cpu = cache[donor_index]
    while True:
        section = int(rng.integers(5, len(order) - 5))
        tissue = torch.nonzero(volume_cpu[0, 0, section].flatten() > .08).flatten().numpy()
        if len(tissue):
            break
    row, column = divmod(int(rng.choice(tissue)), 192)
    tissue_point = (center[0] + section * step
        + (column / 192 - .5) * edge[:, 0] + (row / 192 - .5) * edge[:, 1])
    rotation = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
    spans = np.exp(rng.uniform(math.log(8500), math.log(15500), 2)).astype(np.float32)
    plane_center = tissue_point + rotation[:, :2] @ (rng.uniform(-.18, .18, 2) * spans)
    reflection = int(rng.integers(2))
    y, x = np.meshgrid(np.arange(side) / side, np.arange(side) / side, indexing='ij')
    if reflection:
        x = (side - 1) / side - x
    plane = (plane_center + (x[..., None] - .5) * rotation[:, 0] * spans[0]
        + (y[..., None] - .5) * rotation[:, 1] * spans[1])
    thickness_um = float(rng.uniform(25, 100))
    offsets = np.linspace(-thickness_um / 2, thickness_um / 2, 9, dtype=np.float32)
    weights = np.array([1, 2, 2, 2, 2, 2, 2, 2, 1], np.float32) / 16
    world = plane[None] + offsets[:, None, None, None] * rotation[:, 2]
    donor_points = (world - center[0]) @ inverse.T
    grid = np.stack((((donor_points[..., 0] + .5) * 192 / 191) * 2 - 1,
        ((donor_points[..., 1] + .5) * 192 / 191) * 2 - 1,
        donor_points[..., 2] / (len(order) - 1) * 2 - 1), -1)
    grid = torch.from_numpy(grid[None].astype(np.float32)).to(device)
    volume = volume_cpu.to(device)
    sampled = F.grid_sample(volume, grid, align_corners=True)[0, 0]
    support = F.grid_sample((volume > .04).float(), grid, align_corners=True)[0, 0]
    weights_t = torch.from_numpy(weights).to(device)
    image = (sampled * weights_t[:, None, None]).sum(0)
    support = (support * weights_t[:, None, None]).sum(0)
    donor_image = image

    mode = ('raw', 'exact_black', 'imperfect_brush')[int(rng.choice(3, p=MODE_PROBS))]
    background_mean = float(rng.uniform(0, .8))
    background_slope = rng.uniform(-.15, .15, 2).tolist()
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, side, device=device),
        torch.linspace(-1, 1, side, device=device), indexing='ij')
    background = (background_mean + background_slope[0] * gy
        + background_slope[1] * gx).clamp(0, 1)
    gain = float(rng.uniform(.7, 1.3))
    gamma = float(np.exp(rng.uniform(np.log(.7), np.log(1.4))))
    image = (image.clamp(0, 1).pow(gamma) * gain
        + (1 - support).clamp(0, 1) * background).clamp(0, 1)
    mask = support > .15
    radius = int(rng.integers(1, 6))
    if mode == 'imperfect_brush':
        if rng.integers(2):
            mask = F.max_pool2d(mask[None, None].float(), 2 * radius + 1,
                1, radius)[0, 0] > 0
        else:
            mask = -F.max_pool2d(-F.pad(mask[None, None].float(),
                (radius,) * 4), 2 * radius + 1, 1)[0, 0] > 0
    inputs = torch.zeros(1, 5, side, side, device=device)
    inputs[0, 0] = image
    pose = full_frame_state_from_components(
        torch.from_numpy(plane_center[None].astype(np.float32)).to(device),
        torch.from_numpy(rotation[None]).to(device),
        torch.diag(torch.from_numpy(spans)).to(device)[None])
    source_ids = [donor['identities'][i]['section_id'] for i in order]
    lo = max(0, int(np.floor(donor_points[..., 2].min())))
    hi = min(len(order) - 1, int(np.ceil(donor_points[..., 2].max())))
    touched = [donor['identities'][order[i]] for i in range(lo, hi + 1)]
    anchor = donor['identities'][order[section]]
    records_path = donor['images'].parent / 'records.jsonl'
    provenance = {**anchor,
        'section_id': f"{anchor['animal_id']}-virtual-oblique-{int(seed):016d}",
        'mode': mode,
        'appearance': {'background_mean': background_mean,
            'background_slope_yx': background_slope, 'tissue_gain': gain,
            'tissue_gamma': gamma, 'brush_radius_px': radius},
        'real_oblique_v3': {'seed': int(seed), 'donor_index': donor_index,
            'anchor_section_id': anchor['section_id'],
            'source_section_count': len(order),
            'source_section_ids_sha256': hashlib.sha256(
                json.dumps(source_ids).encode()).hexdigest(),
            'source_images_sha256': context['bindings'][str(donor['images'])],
            'source_records_sha256': context['bindings'][str(records_path)],
            'source_sorted_index_span': [lo, hi],
            'source_span_role': 'conservative whole-raster potential interpolation sources',
            'source_specimen_ids': sorted({r['specimen_id'] for r in touched}),
            'source_experiment_ids': sorted({r['experiment_id'] for r in touched}),
            'plane_center_ap_dv_ml_um': plane_center.tolist(),
            'plane_rotation': rotation.tolist(), 'spans_um': spans.tolist(),
            'reflection': reflection, 'thickness_um': thickness_um,
            'virtual_cut': True,
            'reference_role': 'weak Allen affine; serial TRAIN donor reslice, not acquired oblique histology'}}
    source = {'inputs': inputs, 'state': pose,
        'reflection': torch.tensor([reflection], device=device),
        'offsets': torch.from_numpy(offsets[None]).to(device),
        'weights': weights_t[None],
        'centre': torch.from_numpy(plane[None].astype(np.float32)).to(device),
        'visible': support[None], 'support': support[None],
        'masks': mask[None],
        'eligible': torch.tensor([int((support > .2).sum()) >= side * side * .05],
            device=device), 'provenance': [provenance]}
    sample = sample_one_shot_slide_artifacts_v3(context, [donor_index], seed,
        side=side, source_sample=source)
    if return_source:
        sample['source_image'] = donor_image[None]
        sample['source_support'] = support[None]
        sample['source_state'] = pose
        sample['source_centre'] = source['centre']
    return sample
