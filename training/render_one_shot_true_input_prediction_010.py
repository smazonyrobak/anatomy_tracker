"""Show real generator outputs beside source and predicted atlas planes."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates
import torch

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
evaluation = root / 'runs/one_shot_fitted_ranking_010_development_eval'
checkpoint_path = root / 'runs/one_shot_fitted_ranking_010/joint_step_12000.pt'
out = root / 'reports/one_shot_true_input_prediction_20261002'
torch.set_num_threads(4)
records = {r['section_id']: r for r in map(json.loads, (panel / 'records.jsonl').open())}
rows = [r for r in map(json.loads, (evaluation / 'rows.jsonl').open())
        if r['set'] == 'synthetic' and r['eligible']]
near = [r for r in rows if r['appearance_mode'] == 'raw'
        and abs(records[r['section_id']]['plane_normal_ap_dv_ml'][0]) >= .8]
oblique = [r for r in rows if r['appearance_mode'] == 'exact_black'
           and abs(records[r['section_id']]['plane_normal_ap_dv_ml'][0]) < .4]
chosen = [min(near, key=lambda r: (abs(r['fitted_tissue_um']
                                    - np.median([a['fitted_tissue_um'] for a in near])),
                                   r['section_id'])),
          min(oblique, key=lambda r: (abs(r['fitted_tissue_um']
                                       - np.median([a['fitted_tissue_um'] for a in oblique])),
                                      r['section_id'])),
          max(rows, key=lambda r: r['fitted_tissue_um'])]
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 12000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True).eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
atlas, _ = _decode_and_preprocess_allen_v6()
axis = np.arange(256, dtype=np.float32) / 256
yy, xx = np.meshgrid(axis, axis, indexing='ij')


def render(state, reflection, offsets, weights):
    centre, frame, basis = full_frame_state_to_components(torch.as_tensor(state).float())
    s = 255 / 256 - xx if reflection else xx
    chart = np.stack((s, yy), -1)
    plane = centre.numpy() + (chart - .5) @ (frame[:, :2] @ basis).numpy().T
    normal = frame[:, 2].numpy()
    intensity = np.zeros((256, 256), dtype=np.float32)
    support = np.zeros_like(intensity)
    weights = weights / weights.sum()
    for axial_offset, weight in zip(offsets, weights):
        query = np.moveaxis((plane + axial_offset * normal) / 25 - .5, -1, 0)
        intensity += weight * map_coordinates(atlas[0], query, order=1, mode='constant')
        support += weight * map_coordinates(atlas[1], query, order=1, mode='constant')
    image = np.clip(intensity / np.maximum(support, 1e-4), 0, 1)
    image[support < .05] = 0
    return Image.fromarray((image * 255).astype(np.uint8)).convert('RGB')


tile, gap, left, top = 256, 14, 24, 78
canvas = Image.new('RGB', (left * 2 + tile * 3 + gap * 2,
                           top + 3 * (tile + 66)), (238, 238, 238))
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=19)
small = ImageFont.load_default(size=16)
for column, title in enumerate(('True source atlas plane', 'Synthetic model input',
                                'Model-selected atlas plane')):
    draw.text((left + column * (tile + gap), 22), title, font=font, fill=(20, 20, 20))

selection = []
for row_index, (record, label) in enumerate(zip(chosen, ('near-coronal median',
                                                        'strong-oblique median',
                                                        'worst eligible failure'))):
    metadata = records[record['section_id']]
    with np.load(panel / metadata['file'], allow_pickle=False) as arrays:
        observed = Image.fromarray((np.clip(arrays['inputs'][0], 0, 1) * 255).astype(np.uint8)).convert('RGB')
        source_state = metadata['provenance']['one_shot']['source_state']
        reflection = bool(arrays['reflection'])
        offsets, weights = arrays['offsets_um'].copy(), arrays['weights'].copy()
        inputs = torch.from_numpy(arrays['inputs'][None].copy())
    with torch.inference_mode():
        prediction = model.predict(inputs)
    branch = int(record['fitted_choice'])
    predicted_state = prediction['state'][0, branch // 2]
    true_plane = render(source_state, reflection, offsets, weights)
    predicted_plane = render(predicted_state, bool(branch % 2), offsets, weights)
    y0 = top + row_index * (tile + 66)
    draw.text((left, y0 - 32), f'{label} | {metadata["appearance_mode"].replace("_", " ")} | '
              f'{record["valid_pixels"] / 65536:.0%} tissue | '
              f'{record["fitted_tissue_um"] / 1000:.2f} mm error',
              font=small, fill=(20, 20, 20))
    for column, image in enumerate((true_plane, observed, predicted_plane)):
        canvas.paste(image, (left + column * (tile + gap), y0))
    selection.append({'section_id': metadata['section_id'], 'animal_id': metadata['animal_id'],
                      'specimen_id': metadata['specimen_id'],
                      'experiment_id': metadata['experiment_id'],
                      'synthetic_source_sha256': metadata['sha256'],
                      'appearance_mode': metadata['appearance_mode'],
                      'visible_pixels': record['valid_pixels'],
                      'fitted_choice': branch, 'fitted_tissue_um': record['fitted_tissue_um']})

out.mkdir(parents=True, exist_ok=False)
image_path = out / 'true_input_predicted_atlas_planes.png'
canvas.save(image_path)
(out / 'selection.json').write_text(json.dumps({
    'selection_rule': 'median selected rigid tissue error among eligible raw near-coronal; '
                      'median among eligible exact-black strongly oblique; worst eligible overall',
    'true_plane': 'generator source_state and reflection, finite-thickness atlas render',
    'observed': 'saved actual synthetic input channel 0 including deformation, artifacts and background',
    'predicted_plane': 'frozen 010 fitted-choice state/reflection, finite-thickness atlas render before local map',
    'error': 'mean 3D tissue-coordinate error on visible pixels before local map',
    'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    'evaluation_rows_sha256': hashlib.sha256((evaluation / 'rows.jsonl').read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
    'selection': selection}, indent=2))
print(image_path)
