"""TRAIN-only interior texture comparison; a tissue proxy is not a segmentation label."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

source = Path(__file__).resolve()
real = root / 'data/joint_v7_reserved_train_images_192_001'
prior = root / 'runs/reserved_real_train_appearance_census_001'
out = root / 'runs/v3_interior_texture_119_train'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
prior_done = json.loads((prior / 'completed.json').read_text())
assert hashlib.sha256((prior / 'rows.jsonl').read_bytes()).hexdigest() == prior_done['rows_sha256']
real_rows = [json.loads(line) for line in (prior / 'rows.jsonl').open()]
manifest = json.loads((real / 'completed.json').read_text())
assert len(real_rows) == 512 and len({row['donor_id'] for row in real_rows}) == 512
assert manifest['training_donors'] == 1885 and manifest['training_images'] == 263754
samples = []
for row in real_rows:
    donor = row['donor_id']
    directory = real / f'donor_{donor}'
    assert hashlib.sha256((directory / 'completed.json').read_bytes()).hexdigest() == row['donor_receipt_sha256']
    image = np.asarray(np.load(directory / 'images.npy', mmap_mode='r')[row['array_row_index'], 0], dtype=np.float32)
    samples.append(({'domain': 'real', 'animal_id': donor, 'specimen_id': row['specimen_id'],
        'experiment_id': row['experiment_id'], 'section_id': row['section_id'],
        'donor_receipt_sha256': row['donor_receipt_sha256']}, image.copy(), None))

context = load_streaming_synthetic_v7_64(device='cuda')
rng = np.random.default_rng(2026100919)
attempts, synthetic_count, draw_seed = 0, 0, 2026100920
with torch.inference_mode():
    while synthetic_count < 256:
        virtual = rng.integers(len(context['subjects']), size=8).tolist()
        batch = sample_one_shot_slide_artifacts_v3(context, virtual, draw_seed, side=256)
        images = F.interpolate(batch['inputs'][:, :1], (192, 192), mode='area')[:, 0].cpu().numpy()
        validity = (F.interpolate(batch['valid_mask'][:, None].float(), (192, 192), mode='area')
            [:, 0] > .999).cpu().numpy()
        for i, provenance in enumerate(batch['provenance']):
            attempts += 1
            if not bool(batch['eligible'][i]) or synthetic_count == 256:
                continue
            artifact = provenance['one_shot_slide_artifacts_v3']
            samples.append(({'domain': 'synthetic', 'animal_id': provenance['base_lineage']['animal_id'],
                'specimen_id': provenance['base_lineage']['specimen_id'],
                'experiment_id': provenance['base_lineage']['experiment_id'],
                'section_id': provenance['physical_section_id'],
                'virtual_subject_id': provenance['virtual_subject_id'],
                'virtual_index': provenance['virtual_index'],
                'mode': provenance['mode'], 'raw_exterior_code': artifact['parameters']['raw_exterior_code'],
                'events': artifact['events'], 'draw_seed': draw_seed, 'draw_row': i},
                images[i].copy(), validity[i].copy()))
            synthetic_count += 1
        draw_seed += 1
assert len({meta['section_id'] for meta, _, _ in samples if meta['domain'] == 'synthetic'}) == 256

rows = []
for meta, image, valid in samples:
    foreground = ndi.binary_fill_holes(ndi.binary_closing(image > .03, iterations=3))
    labels, count = ndi.label(foreground)
    tissue = labels == np.bincount(labels.ravel())[1:].argmax() + 1 if count else foreground
    interior = ndi.binary_erosion(tissue, iterations=6)
    row = {**meta, 'proxy_tissue_fraction': float(tissue.mean()),
        'interior_fraction': float(interior.mean()), 'interior_pixels': int(interior.sum())}
    if valid is not None:
        row['synthetic_valid_fraction'] = float(valid.mean())
        row['proxy_precision_against_valid'] = float((tissue & valid).sum() / max(1, tissue.sum()))
        row['proxy_recall_against_valid'] = float((tissue & valid).sum() / max(1, valid.sum()))
    if int(interior.sum()) >= 1024:
        values = image[interior]
        contrast = float(np.quantile(values, .9) - np.quantile(values, .1))
        high1 = image - ndi.gaussian_filter(image, 1)
        high3 = image - ndi.gaussian_filter(image, 3)
        row.update(interior_contrast_q90_q10=contrast,
            interior_highpass_1px_rms=float(np.sqrt(np.mean(high1[interior] ** 2))),
            interior_highpass_3px_rms=float(np.sqrt(np.mean(high3[interior] ** 2))),
            interior_highpass_1px_over_contrast=float(np.sqrt(np.mean(high1[interior] ** 2)) / max(.01, contrast)),
            interior_highpass_3px_over_contrast=float(np.sqrt(np.mean(high3[interior] ** 2)) / max(.01, contrast)),
            interior_dark_fraction_below_0_1=float((values < .1).mean()),
            interior_bright_fraction_above_0_9=float((values > .9).mean()))
    rows.append(row)

metrics = ('proxy_tissue_fraction', 'interior_fraction', 'interior_contrast_q90_q10',
    'interior_highpass_1px_rms', 'interior_highpass_3px_rms',
    'interior_highpass_1px_over_contrast', 'interior_highpass_3px_over_contrast',
    'interior_dark_fraction_below_0_1', 'interior_bright_fraction_above_0_9')
groups = {'real': [r for r in rows if r['domain'] == 'real'],
    'synthetic_raw_black': [r for r in rows if r['domain'] == 'synthetic'
        and r['mode'] == 'raw' and r['raw_exterior_code'] == 0],
    'synthetic_other': [r for r in rows if r['domain'] == 'synthetic'
        and not (r['mode'] == 'raw' and r['raw_exterior_code'] == 0)]}
summary = {'scope': '512 donor-distinct Allen TRAIN sections versus 256 fresh independent v3 synthetic TRAIN planes; descriptive only',
    'source': '192x192 real and area-downsampled 256->192 synthetic; fixed image-threshold morphological tissue proxy, not biological truth',
    'synthetic_attempts': attempts, 'synthetic_eligible': synthetic_count,
    'groups': {name: {'sections': len(group), 'measurable': sum('interior_contrast_q90_q10' in r for r in group),
        'metrics': {key: {'mean': float(np.mean([r[key] for r in group if key in r])),
            'q10_q50_q90': np.quantile([r[key] for r in group if key in r], [.1, .5, .9]).tolist()}
            for key in metrics if any(key in r for r in group)}} for name, group in groups.items()},
    'limitations': 'Threshold tissue proxy can miss dark tissue; frequency/contrast scalars do not validate anatomy, artifact realism or target-lab prevalence.'}
out.mkdir(parents=True, exist_ok=False)
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'real_sections': 512, 'synthetic_sections': 256,
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'generator_source_sha256': {name: hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
        for name in ('arbitrary_plane_one_shot_slide_artifacts_v3.py',
            'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
            'arbitrary_plane_streaming_synthetic_v7.py',
            'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'synthetic_context_provenance_sha256': hashlib.sha256(json.dumps(
        context['provenance'], sort_keys=True).encode()).hexdigest(),
    'real_manifest_sha256': hashlib.sha256((real / 'completed.json').read_bytes()).hexdigest(),
    'prior_census_rows_sha256': prior_done['rows_sha256'],
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest()}, indent=2))
print(json.dumps({'event': 'complete', 'groups': {name: {'sections': group['sections'],
    'measurable': group['measurable'], 'highpass_1px_over_contrast':
    group['metrics'].get('interior_highpass_1px_over_contrast')} for name, group in summary['groups'].items()}}), flush=True)
