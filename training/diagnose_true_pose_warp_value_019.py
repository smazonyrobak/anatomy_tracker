"""Read-only exact-pose no-warp control against frozen 019 development rows."""
import json
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

root = Path('I:/AnatomyTracker')
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
evaluation = root / 'runs/one_shot_exposure_019_development_eval/rows.jsonl'
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
mapped = {row['sha256']: row['true_mapped_um'] for row in map(json.loads, evaluation.open())
          if row['set'] == 'synthetic' and row['step'] == 18000}
assert len(records) == len(mapped) == 185
rows = []
for row in records:
    with np.load(panel / row['file'], allow_pickle=False) as data:
        state = torch.from_numpy(data['target_state'])
        centre, frame, basis = full_frame_state_to_components(state)
        index = np.flatnonzero(data['valid_mask'])
        target = data['target_centre_um'].reshape(-1, 3)[index]
        chart = np.stack((index % 256, index // 256), -1).astype(np.float32) / 256
        if bool(data['reflection']):
            chart[:, 0] = 255 / 256 - chart[:, 0]
    rigid = centre.numpy() + (chart - .5) @ (frame[:, :2] @ basis).numpy().T
    rigid_error = np.linalg.norm(rigid - target, axis=-1).mean()
    rows.append({'animal_id': row['animal_id'], 'rigid_um': rigid_error,
                 'mapped_um': mapped[row['sha256']],
                 'warp_strength': row['provenance']['one_shot']['warp_strength'],
                 'events': row['provenance']['one_shot']['events']})
animals = sorted({row['animal_id'] for row in rows})
result = {name: float(np.mean([np.mean([row[name] for row in rows if row['animal_id'] == animal])
                               for animal in animals])) for name in ('rigid_um', 'mapped_um')}
result['mapped_better_fraction'] = float(np.mean([row['mapped_um'] < row['rigid_um'] for row in rows]))
result['identity_equal_gain_um'] = result['rigid_um'] - result['mapped_um']
result['events'] = {event: {'n': int(sum(row['events'][event] for row in rows)),
                            'mean_gain_um': float(np.mean([row['rigid_um'] - row['mapped_um']
                                                           for row in rows if row['events'][event]]))}
                    for event in ('tear', 'missing', 'fold', 'bubble', 'tile_seam')}
threshold = np.quantile([row['warp_strength'] for row in rows], .75)
result['top_warp_quartile'] = {'n': int(sum(row['warp_strength'] >= threshold for row in rows)),
                              'mean_gain_um': float(np.mean([row['rigid_um'] - row['mapped_um']
                                                             for row in rows if row['warp_strength'] >= threshold]))}
print(json.dumps(result))
