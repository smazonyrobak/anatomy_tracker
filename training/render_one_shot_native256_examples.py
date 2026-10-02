"""Show frozen generator sections beside their unwarped target atlas planes."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf

root = Path('I:/AnatomyTracker/data/one_shot_native256_synthetic_dev_001')
rows = [json.loads(line) for line in (root / 'records.jsonl').open()]
selection = [('raw', 'bubble'), ('exact_black', 'tear'),
             ('imperfect_brush', 'missing'), ('raw', 'fold'),
             ('exact_black', 'tile_seam')]
selected = [max((row for row in rows if row['eligible'] and row['appearance_mode'] == mode
                 and row['provenance']['one_shot']['events'][event]),
                key=lambda row: row['valid_pixels']) for mode, event in selection]
selected.append(min((row for row in rows if row['eligible']
                     and row['appearance_mode'] == 'imperfect_brush'
                     and row['file'] not in {item['file'] for item in selected}),
                    key=lambda row: row['valid_pixels']))
atlas, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas)
side = 256
y, x = torch.meshgrid(torch.arange(side) / side, torch.arange(side) / side, indexing='ij')
fig, axes = plt.subplots(len(selected), 2, figsize=(8, 20), constrained_layout=True)
with torch.no_grad():
    for index, (axis, row) in enumerate(zip(axes, selected)):
        with np.load(root / row['file']) as sample:
            state = torch.from_numpy(sample['target_state'][None])
            reflection = int(sample['reflection'])
            offsets = torch.from_numpy(sample['offsets_um'][None])
            weights = torch.from_numpy(sample['weights'][None])
            observed = sample['inputs'][0]
        center, frame, basis = full_frame_state_to_components(state)
        sx = (side - 1) / side - x if reflection else x
        chart = torch.stack((sx, y), -1)[None]
        plane = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        coordinates = (plane[:, None] + offsets[:, :, None, None, None]
                       * frame[:, None, None, None, :, 2])
        rendered = render_finite_thickness_coordinate_grid(
            atlas, coordinates, (0., 0., 0.), (25., 25., 25.), weights)[0].numpy()
        reference = rendered[0] / np.maximum(rendered[1], 1e-4)
        reference[rendered[1] < .05] = 0
        axis[0].imshow(observed, cmap='gray', vmin=0, vmax=1)
        axis[1].imshow(reference, cmap='gray', vmin=0, vmax=1)
        event = selection[index][1] if index < len(selection) else 'small edge section'
        axis[0].set_title(f"Generated slice · {row['appearance_mode']} · {event}")
        axis[1].set_title('True atlas plane · no tissue warp')
        for panel in axis:
            panel.axis('off')
fig.savefig('I:/AnatomyTracker/runs/one_shot_native256_generator_examples.png', dpi=135)
