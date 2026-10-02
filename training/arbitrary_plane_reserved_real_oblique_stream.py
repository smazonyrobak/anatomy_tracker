"""Fresh arbitrary-plane training cuts from one donor's real Allen serial stack.

Coordinates inherit weak Allen section affines; they are not expert ground truth.
Each call draws a new plane before the existing one-shot artifact augmentation.
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
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream


def sample_reserved_real_oblique_train(context, donor_index, seed, side=256, device='cpu'):
    donor = context['donors'][int(donor_index)]
    rng = np.random.default_rng(seed)
    order = np.argsort(donor['state'][:, 0])
    states = torch.from_numpy(donor['state'][order].copy())
    center, frame, basis = full_frame_state_to_components(states)
    center, frame, basis = center.numpy(), frame.numpy(), basis.numpy()
    edge = frame[0, :, :2] @ basis[0]
    step = (center[-1] - center[0]) / (len(order) - 1)
    assert np.max(np.linalg.norm(center - (center[0] + np.arange(len(order))[:, None] * step), axis=1)) < 5
    assert np.max(np.abs(states.numpy()[:, 3:] - states.numpy()[0, 3:])) < 1e-5
    inverse = np.linalg.inv(np.column_stack((edge[:, 0], edge[:, 1], step)))
    images = np.load(donor['images'], mmap_mode='r')
    volume = torch.from_numpy(images[order, :1].astype(np.float32).transpose(1, 0, 2, 3))[None]
    while True:
        section = int(rng.integers(5, len(order) - 5))
        tissue = np.flatnonzero(images[order[section], 0] > .08)
        if len(tissue):
            break
    pixel = int(rng.choice(tissue))
    row, column = divmod(pixel, 192)
    tissue_point = (center[0] + section * step
                    + (column / 192 - .5) * edge[:, 0]
                    + (row / 192 - .5) * edge[:, 1])
    rotation = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
    spans = np.exp(rng.uniform(math.log(8500), math.log(15500), 2)).astype(np.float32)
    plane_center = (tissue_point + rotation[:, :2]
                    @ (rng.uniform(-.18, .18, 2) * spans))
    reflection = int(rng.integers(2))
    y, x = np.meshgrid(np.arange(side) / side, np.arange(side) / side, indexing='ij')
    if reflection:
        x = (side - 1) / side - x
    plane = (plane_center + (x[..., None] - .5) * rotation[:, 0] * spans[0]
             + (y[..., None] - .5) * rotation[:, 1] * spans[1])
    offsets = np.linspace(-25., 25., 9, dtype=np.float32)
    weights = np.ones(9, dtype=np.float32)
    weights[[0, -1]] = .5
    weights /= weights.sum()
    world = plane[None] + offsets[:, None, None, None] * rotation[:, 2]
    donor_points = (world - center[0]) @ inverse.T
    grid = np.stack(((donor_points[..., 0] + .5) * 192 / 191 * 2 - 1,
                     (donor_points[..., 1] + .5) * 192 / 191 * 2 - 1,
                     donor_points[..., 2] / (len(order) - 1) * 2 - 1), -1)
    grid = torch.from_numpy(grid[None].astype(np.float32))
    sampled = F.grid_sample(volume, grid, align_corners=True)[0, 0]
    support = F.grid_sample((volume > .04).float(), grid, align_corners=True)[0, 0]
    weights_t = torch.from_numpy(weights)
    image = (sampled * weights_t[:, None, None]).sum(0)
    support = (support * weights_t[:, None, None]).sum(0)
    valid = support > .2
    inputs = torch.zeros(1, 5, side, side)
    inputs[0, 0] = image
    pose = full_frame_state_from_components(
        torch.from_numpy(plane_center[None].astype(np.float32)),
        torch.from_numpy(rotation[None]), torch.diag(torch.from_numpy(spans))[None])
    central_identity = donor['identities'][order[section]]
    source_ids = [donor['identities'][i]['section_id'] for i in order]
    source_span = [max(0, int(np.floor(donor_points[..., 2].min()))),
                   min(len(order) - 1, int(np.ceil(donor_points[..., 2].max())))]
    provenance = {**central_identity,
                  'section_id': f"{central_identity['animal_id']}-oblique-reslice-{int(seed):016d}",
                  'mode': 'raw',
                  'appearance': {'background_mean': 0., 'background_slope_yx': [0., 0.]},
                  'real_oblique': {'seed': int(seed), 'donor_index': int(donor_index),
                                   'source_central_section_id': central_identity['section_id'],
                                   'source_section_count': len(order),
                                   'source_section_ids_sha256': hashlib.sha256(
                                       json.dumps(source_ids).encode()).hexdigest(),
                                   'source_sorted_index_span': source_span,
                                   'reference_role': 'weak Allen affine; resliced real serial anatomy'}}
    source = {'inputs': inputs.to(device), 'state': pose.to(device),
              'reflection': torch.tensor([reflection], device=device),
              'offsets': torch.from_numpy(offsets[None]).to(device),
              'weights': weights_t[None].to(device),
              'centre': torch.from_numpy(plane[None].astype(np.float32)).to(device),
              'visible': support[None].to(device), 'support': support[None].to(device),
              'masks': torch.ones(1, side, side, dtype=torch.bool, device=device),
              'eligible': torch.tensor([int(valid.sum()) >= side * side / 144], device=device),
              'provenance': [provenance]}
    return sample_one_shot_stream(None, [int(donor_index)], seed, side=side,
                                  source_sample=source)
