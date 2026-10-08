"""Small TRAIN-only QC grid from the actual slide-artifact v2 generator."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from training.arbitrary_plane_one_shot_slide_artifacts_v2 import (
    VERSION, sample_one_shot_slide_artifacts_v2,
)
from training.arbitrary_plane_streaming_synthetic_v7 import (
    load_streaming_synthetic_v7, sample_streaming_synthetic_v7,
)


out = Path('I:/AnatomyTracker/reports/one_shot_slide_artifacts_v2_qc_20261008_002')
torch.set_num_threads(4)
pool = []
attempts = []
rng = np.random.default_rng(20261008)
with torch.inference_mode():
    context = load_streaming_synthetic_v7(device='cuda')
    for draw in range(8):
        subjects = rng.integers(len(context['subjects']), size=3).tolist()
        seed = 2026100800 + draw
        source = sample_streaming_synthetic_v7(context, subjects, seed, side=192)
        batch = sample_one_shot_slide_artifacts_v2(context, subjects, seed, side=192, source_sample=source)
        for row, record in enumerate(batch['provenance']):
            eligible = bool(batch['eligible'][row])
            attempts.append({'seed': seed, 'row': row, 'virtual_index': subjects[row],
                             'physical_section_id': record['physical_section_id'], 'eligible': eligible})
            if eligible:
                atlas = (source['clean'][row] / source['support'][row].clamp_min(1e-4)).clamp(0, 1)
                atlas = torch.where(source['support'][row] > .05, atlas, 0)
                pool.append({'record': record, 'observed': batch['inputs'][row, 0].cpu().numpy(),
                    'atlas_truth': atlas.cpu().numpy(), 'centre': batch['centre'][row].cpu().numpy(),
                    'valid': batch['valid_mask'][row].cpu().numpy(),
                    'fragment_displaced': batch['fragment_displaced_mask'][row].cpu().numpy(),
                    'fragment_overlap': batch['fragment_overlap_mask'][row].cpu().numpy(),
                    'fold_occlusion': batch['fold_occlusion_mask'][row].cpu().numpy(),
                    'fold_overlap': batch['fold_overlap_mask'][row].cpu().numpy(),
                    'bubble_invalid': batch['bubble_invalid_mask'][row].cpu().numpy(),
                    'state': batch['state'][row].cpu().numpy(),
                    'reflection': batch['reflection'][row].cpu().numpy(),
                    'offsets': batch['offsets'][row].cpu().numpy(),
                    'weights': batch['weights'][row].cpu().numpy()})
        if (len(pool) >= 6 and any(item['record'][VERSION]['events']['bubble'] for item in pool)
                and any(item['record']['mode'] == 'raw' for item in pool)
                and any(item['record'][VERSION]['events']['tile_seam'] for item in pool)):
            break
selected = []
for kind, name in (('event', 'bubble'), ('mode', 'raw'), ('event', 'tile_seam'),
                   ('event', 'fold'), ('event', 'fragment')):
    for index, item in enumerate(pool):
        matches = (item['record']['mode'] == name if kind == 'mode'
                   else item['record'][VERSION]['events'][name])
        if matches and index not in selected:
            selected.append(index)
            break
selected += [index for index in range(len(pool)) if index not in selected]
chosen = [pool[index] for index in selected[:6]]
assert len(chosen) == 6
assert any(item['record'][VERSION]['events']['bubble'] for item in chosen)

out.mkdir(parents=True, exist_ok=False)
tile, gap, label = 192, 8, 57
cell_w, cell_h = 3 * tile + 2 * gap, tile + label
sheet = Image.new('RGB', (3 * cell_w + 4 * 12, 2 * cell_h + 3 * 12), (235, 235, 235))
draw = ImageDraw.Draw(sheet)
artifacts = []
for index, item in enumerate(chosen):
    observed = np.uint8(np.rint(np.clip(item['observed'], 0, 1) * 255))
    atlas = np.uint8(np.rint(np.clip(item['atlas_truth'], 0, 1) * 255))
    mask = np.zeros((tile, tile, 3), dtype=np.uint8)
    mask[item['valid']] = (230, 230, 230)
    mask[item['fragment_displaced'] & item['valid']] = (35, 185, 215)
    mask[item['fragment_overlap']] = (225, 30, 175)
    mask[item['fold_occlusion']] = (230, 65, 35)
    mask[item['bubble_invalid']] = (235, 190, 35)
    x0 = 12 + (index % 3) * (cell_w + 12)
    y0 = 12 + (index // 3) * (cell_h + 12)
    events = ', '.join(key for key, enabled in item['record'][VERSION]['events'].items() if enabled) or 'clean'
    draw.text((x0, y0), f"{index + 1}: {item['record']['mode']} | {events}", fill=(20, 20, 20))
    for column, (name, image) in enumerate((('observed', Image.fromarray(observed, 'L')),
                                            ('atlas truth', Image.fromarray(atlas, 'L')),
                                            ('valid CCF / defects', Image.fromarray(mask, 'RGB')))):
        x = x0 + column * (tile + gap)
        draw.text((x, y0 + 20), name, fill=(25, 25, 25))
        sheet.paste(image.convert('RGB'), (x, y0 + label))
    path = out / f'sample_{index + 1:02d}.npz'
    np.savez_compressed(path, **{key: value for key, value in item.items() if key != 'record'})
    artifacts.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'provenance': item['record']})
grid_path = out / 'qc_grid.png'
sheet.save(grid_path)
receipt = {'generator_version': VERSION, 'split': 'TRAIN synthetic only', 'not_model_predictions': True,
    'atlas_truth': 'unwarped source finite-thickness atlas render through the TRAIN virtual-subject inverse map; not an affine approximation or model prediction',
    'mask_colours': {'white': 'valid stationary tissue', 'cyan': 'valid displaced single-layer fragment',
                     'magenta': 'invalid fragment overlap', 'red': 'invalid folded tissue/occlusion',
                     'yellow': 'invalid bubble defocus', 'black': 'other invalid/background'},
    'selection': 'six distinct eligible independently drawn planes, prioritizing bubble, raw, seam, fold and fragment coverage; all attempts logged',
    'attempted_rows': attempts, 'context_input_sha256': context['provenance']['input_sha256'],
    'context_source_sha256': context['provenance']['source_sha256'],
    'context_provenance_sha256': hashlib.sha256(json.dumps(context['provenance'], sort_keys=True).encode()).hexdigest(),
    'generator_sha256': hashlib.sha256(Path(sample_one_shot_slide_artifacts_v2.__code__.co_filename).read_bytes()).hexdigest(),
    'qc_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'grid': grid_path.name, 'grid_sha256': hashlib.sha256(grid_path.read_bytes()).hexdigest(),
    'artifacts': artifacts}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'grid': str(grid_path), 'samples': len(chosen), 'attempted': len(attempts),
                  'event_counts': {key: sum(r['provenance'][VERSION]['events'][key] for r in artifacts)
                                   for key in ('fragment', 'fold', 'bubble', 'tile_seam')}}))
