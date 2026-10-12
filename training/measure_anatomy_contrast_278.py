"""Small TRAIN-only v11 texture and exterior census using the 119 tissue proxy."""
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
import torch
from scipy import ndimage as ndi

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_slide_artifacts_v11 import sample_one_shot_slide_artifacts_v11
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64


source = Path(__file__).resolve()
preview = root / 'previews/anatomy_contrast_278_actual_generator_001'
prior = root / 'runs/v3_interior_texture_119_train'
out = root / 'runs/anatomy_contrast_278_texture_census_001'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
preview_receipt = json.loads((preview / 'completed.json').read_text())
prior_receipt = json.loads((prior / 'completed.json').read_text())
assert hashlib.sha256((preview / 'selection.json').read_bytes()).hexdigest() == preview_receipt['selection_sha256']
assert hashlib.sha256((prior / 'summary.json').read_bytes()).hexdigest() == prior_receipt['summary_sha256']
reference = json.loads((prior / 'summary.json').read_text())
context = load_streaming_synthetic_v7_64(device='cuda')
atlas_array, annotation_array = _decode_and_preprocess_allen_v6()
annotation = torch.from_numpy(annotation_array.astype(np.int64)).cuda()
del atlas_array, annotation_array
rng = np.random.default_rng(2026101227801)
rows, attempts = [], 0
with torch.inference_mode():
    while len(rows) < 48:
        attempts += 1
        base = int(rng.integers(56))
        subject = base * VARIANTS + int(rng.integers(VARIANTS))
        seed = 2026101227801 + 1_000_000 + attempts
        sample = sample_one_shot_slide_artifacts_v11(context, annotation, [subject], seed, 192)
        if not bool(sample['eligible'][0]):
            continue
        record = sample['provenance'][0]
        image = sample['inputs'][0, 0].cpu().numpy()
        visible = sample['visible'][0].cpu().numpy()
        brush = sample['brush_mask'][0].cpu().numpy()
        exterior = record['one_shot_slide_artifacts_v11']['exterior']['style']
        exterior_violation = (int(np.count_nonzero(image[visible <= .05]))
                              if exterior == 'exact_black' else
                              int(np.count_nonzero(image[~brush]))
                              if exterior == 'exact_black_brush' else None)
        foreground = ndi.binary_fill_holes(ndi.binary_closing(image > .03, iterations=3))
        labels, count = ndi.label(foreground)
        tissue = labels == np.bincount(labels.ravel())[1:].argmax() + 1 if count else foreground
        interior = ndi.binary_erosion(tissue, iterations=6)
        normal = full_frame_state_to_components(sample['state'][0])[1][:, 2]
        row = {'physical_section_id': record['physical_section_id'], 'base_index': base,
               'subject_index': subject, 'seed': seed,
               'obliquity_deg': float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1)))),
               'exterior_style': exterior, 'exterior_nonzero_violation_pixels': exterior_violation,
               'source_tissue_label_fraction': record['one_shot_slide_artifacts_v11']['source_tissue_label_fraction'],
               'proxy_tissue_fraction': float(tissue.mean()),
               'interior_pixels': int(interior.sum())}
        if row['interior_pixels'] >= 1024:
            values = image[interior]
            contrast = float(np.quantile(values, .9) - np.quantile(values, .1))
            high1 = image - ndi.gaussian_filter(image, 1)
            high3 = image - ndi.gaussian_filter(image, 3)
            row.update(interior_contrast_q90_q10=contrast,
                       interior_highpass_1px_over_contrast=float(np.sqrt(np.mean(high1[interior] ** 2)) / max(.01, contrast)),
                       interior_highpass_3px_over_contrast=float(np.sqrt(np.mean(high3[interior] ** 2)) / max(.01, contrast)),
                       interior_dark_fraction_below_0_1=float((values < .1).mean()),
                       interior_bright_fraction_above_0_9=float((values > .9).mean()))
        rows.append(row)
        del sample
assert len({row['physical_section_id'] for row in rows}) == 48
measured = [row for row in rows if 'interior_highpass_1px_over_contrast' in row]
metrics = ('proxy_tissue_fraction', 'interior_contrast_q90_q10',
           'interior_highpass_1px_over_contrast', 'interior_highpass_3px_over_contrast',
           'interior_dark_fraction_below_0_1', 'interior_bright_fraction_above_0_9')
summary = {'sections': len(rows), 'attempts': attempts, 'measurable': len(measured),
           'obliquity_over_40deg': sum(row['obliquity_deg'] > 40 for row in rows),
           'exterior_violation_pixels': sum(row['exterior_nonzero_violation_pixels'] or 0 for row in rows),
           'mean_source_tissue_label_fraction': float(np.mean([row['source_tissue_label_fraction'] for row in rows])),
           'metrics': {name: float(np.mean([row[name] for row in measured])) for name in metrics},
           'reference_119': {group: {name: reference['groups'][group]['metrics'][name]['mean']
                for name in metrics if name in reference['groups'][group]['metrics']}
                for group in ('real', 'synthetic_raw_black')},
           'interpretation': 'descriptive TRAIN image proxy, not anatomical realism or model performance'}
out.mkdir(parents=True, exist_ok=False)
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'script_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'generator_sha256': hashlib.sha256(source.with_name(
        'arbitrary_plane_one_shot_slide_artifacts_v11.py').read_bytes()).hexdigest(),
    'preview_selection_sha256': preview_receipt['selection_sha256'],
    'reference_119_summary_sha256': prior_receipt['summary_sha256'],
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest()}, indent=2))
print(json.dumps({'event': 'complete', **summary}), flush=True)
