"""Evaluation-only atlas-serial control paired to one frozen real-donor virtual cut."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import (
    VERSION, sample_one_shot_slide_artifacts_v3)


def sample_atlas_serial_control_115(donor_context, atlas_context, frozen,
                                    donor_index, seed, device='cuda'):
    """Change only serial-section appearance, retaining frozen pose/artifact draw."""
    donor_index, seed = int(donor_index), int(seed)
    donor = donor_context['donors'][donor_index]
    record = frozen['provenance'][0]
    geometry = record['real_oblique_v3']
    assert geometry['donor_index'] == donor_index and geometry['seed'] == seed
    assert 'source_image' in frozen and 'source_support' in frozen
    side, native = frozen['inputs'].shape[-1], 192
    order = np.argsort(donor['state'][:, 0])
    state_sha256 = hashlib.sha256(donor['state'][order].tobytes()).hexdigest()
    thickness_sha256 = hashlib.sha256(donor['thickness_um'][order].tobytes()).hexdigest()
    atlas_binding_sha256 = hashlib.sha256(json.dumps(
        atlas_context['provenance']['atlas_binding'], sort_keys=True).encode()).hexdigest()
    key = (donor['animal_id'], geometry['source_images_sha256'], state_sha256,
        thickness_sha256, atlas_binding_sha256, str(torch.device(device)))
    cache = donor_context.setdefault('atlas_serial_control_115_cache', {})
    if key not in cache:
        states = torch.from_numpy(donor['state'][order].copy())
        centre, frame, basis = full_frame_state_to_components(states)
        centre, frame, basis = centre.numpy(), frame.numpy(), basis.numpy()
        edge = frame[0, :, :2] @ basis[0]
        step = (centre[-1] - centre[0]) / (len(order) - 1)
        inverse = np.linalg.inv(np.column_stack((edge[:, 0], edge[:, 1], step)))
        atlas = atlas_context['atlas'].to(device)
        y, x = torch.meshgrid(torch.arange(native, device=device) / native,
            torch.arange(native, device=device) / native, indexing='ij')
        serial = []
        axial = torch.linspace(-.5, .5, 9, device=device)
        masses = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device=device,
            dtype=torch.float32) / 16
        with torch.no_grad():
            for start in range(0, len(order), 4):
                stop = min(start + 4, len(order))
                state = states[start:stop].to(device)
                c, f, b = full_frame_state_to_components(state)
                e = f[:, :, :2] @ b
                plane = (c[:, None, None] + (x[None, ..., None] - .5) * e[:, None, None, :, 0]
                    + (y[None, ..., None] - .5) * e[:, None, None, :, 1])
                thickness = torch.from_numpy(donor['thickness_um'][order[start:stop]].copy()).to(device)
                offsets = thickness[:, None] * axial
                coordinates = plane[:, None] + offsets[:, :, None, None, None] * f[:, None, None, None, :, 2]
                rendered = render_finite_thickness_coordinate_grid(atlas, coordinates,
                    (0., 0., 0.), (25., 25., 25.), masses)
                serial.append((rendered[:, :1] / rendered[:, 1:2].clamp_min(1e-4)).clamp(0, 1))
            cache[key] = {'centre': centre, 'inverse': inverse,
                'volume_cpu': torch.cat(serial, 0).permute(1, 0, 2, 3)[None].cpu()}
    centre, inverse = cache[key]['centre'], cache[key]['inverse']
    volume = cache[key]['volume_cpu'].to(device)
    with torch.no_grad():
        rotation = np.asarray(geometry['plane_rotation'], dtype=np.float32)
        spans = np.asarray(geometry['spans_um'], dtype=np.float32)
        plane_center = np.asarray(geometry['plane_center_ap_dv_ml_um'])
        gy, gx = np.meshgrid(np.arange(side) / side, np.arange(side) / side, indexing='ij')
        if geometry['reflection']:
            gx = (side - 1) / side - gx
        oblique = (plane_center + (gx[..., None] - .5) * rotation[:, 0] * spans[0]
            + (gy[..., None] - .5) * rotation[:, 1] * spans[1])
        assert (torch.from_numpy(oblique[None].astype(np.float32)).to(device)
            - frozen['source_centre']).abs().max() < .01
        offsets = frozen['offsets'][0].detach().cpu().numpy()
        world = oblique[None] + offsets[:, None, None, None] * rotation[:, 2]
        donor_points = (world - centre[0]) @ inverse.T
        grid = np.stack((((donor_points[..., 0] + .5) * native / (native - 1)) * 2 - 1,
            ((donor_points[..., 1] + .5) * native / (native - 1)) * 2 - 1,
            donor_points[..., 2] / (len(order) - 1) * 2 - 1), -1)
        grid = torch.from_numpy(grid[None].astype(np.float32)).to(device)
        sampled = F.grid_sample(volume, grid, align_corners=True)[0, 0]
        sampled_support = F.grid_sample((volume > .04).float(), grid,
            align_corners=True)[0, 0]
        weights = frozen['weights'][0].to(device)
        source_image = (sampled * weights[:, None, None]).sum(0)
        source_support = (sampled_support * weights[:, None, None]).sum(0)

    appearance = record['appearance']
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, side, device=device),
        torch.linspace(-1, 1, side, device=device), indexing='ij')
    background = (appearance['background_mean']
        + appearance['background_slope_yx'][0] * yy
        + appearance['background_slope_yx'][1] * xx).clamp(0, 1)
    image = (source_image.clamp(0, 1).pow(appearance['tissue_gamma'])
        * appearance['tissue_gain'] + (1 - source_support).clamp(0, 1) * background).clamp(0, 1)
    original = {key: value for key, value in record.items() if key != VERSION}
    inputs = torch.zeros_like(frozen['inputs'])
    inputs[0, 0] = image
    source = {'inputs': inputs, 'state': frozen['source_state'],
        'reflection': frozen['reflection'], 'offsets': frozen['offsets'],
        'weights': frozen['weights'], 'centre': frozen['source_centre'],
        'visible': source_support[None], 'support': source_support[None],
        'masks': torch.ones_like(source_support[None], dtype=torch.bool),
        'eligible': torch.tensor([int((source_support > .2).sum()) >= side * side * .05],
            device=device),
        'provenance': [{**original, 'atlas_serial_control_115': {
            'role': 'evaluation-only paired atlas-derived serial control; not a new training example',
            'paired_real_section_id': record['section_id'],
            'donor_index': donor_index, 'seed': seed,
            'original_native_section_shape': [native, native],
            'serial_sections': len(order),
            'serial_renderer': 'nine-sample finite-thickness atlas PSF at each original weak Allen section plane',
            'oblique_reslice': 'same trilinear 3D grid and physical plane as paired real virtual cut',
            'grid_sha256': hashlib.sha256(grid.cpu().numpy().tobytes()).hexdigest(),
            'source_state_sha256': state_sha256,
            'source_thickness_sha256': thickness_sha256,
            'source_images_sha256': geometry['source_images_sha256'],
            'source_records_sha256': geometry['source_records_sha256'],
            'source_section_ids_sha256': geometry['source_section_ids_sha256'],
            'atlas_binding': atlas_context['provenance']['atlas_binding'],
            'appearance': appearance,
            'brush': 'reuse paired real final optional-brush mask after identical v3 artifacts',
            'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}]}
    paired = sample_one_shot_slide_artifacts_v3(donor_context,
        [donor_index], seed, side, source_sample=source)
    paired['inputs'][:, 0] *= frozen['brush_mask']
    paired['inputs'][:, 1:3] = frozen['inputs'][:, 1:3]
    paired['brush_mask'] = frozen['brush_mask']
    assert torch.equal(paired['state'], frozen['state'])
    assert torch.equal(paired['reflection'], frozen['reflection'])
    assert torch.equal(paired['offsets'], frozen['offsets'])
    assert torch.equal(paired['weights'], frozen['weights'])
    assert paired['provenance'][0][VERSION]['events'] == record[VERSION]['events']
    assert paired['provenance'][0][VERSION]['parameters'] == record[VERSION]['parameters']
    paired['source_image'] = source_image[None]
    paired['source_support'] = source_support[None]
    paired['source_state'] = frozen['source_state']
    paired['source_centre'] = frozen['source_centre']
    return paired
