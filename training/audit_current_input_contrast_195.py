"""Compare actual TRAIN fluorescence preprocessing with current synthetic v4 intensity."""

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image

from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

split = json.loads((Path(__file__).with_name('lupori_animal_split_185.json')).read_text())
real = []
with ZipFile(root / 'data/lupori_joint_185/rawData.zip') as archive:
    for animal in split['train']:
        info = ElementTree.fromstring(archive.read(f'{animal}/{animal}-info.xml'))
        valid = {row.findtext('name') for row in info.findall('slices')
                 if row.findtext('valid') == '1'}
        rows = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))['slices']
        rows = [row for row in rows if Path(row['filename']).name.removesuffix('-thumb.png') in valid]
        for row in rows[::max(1, len(rows) // 8)][:8]:
            section = Path(row['filename']).name.removesuffix('-thumb.png')
            rgb = np.asarray(Image.open(BytesIO(archive.read(f'{animal}/{row["filename"]}')))
                .convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
            image = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            image = np.clip(image / max(np.percentile(image, 99), .01), 0, 1)
            mask = np.asarray(Image.open(BytesIO(archive.read(
                f'{animal}/masks/{section}-mask.png'))).convert('L').resize(
                (256, 256), Image.Resampling.NEAREST)) > 127
            real.append({'animal': animal, 'section': section,
                'whole_p99': float(np.percentile(image, 99)),
                'tissue_p50': float(np.median(image[mask])),
                'tissue_p90': float(np.percentile(image[mask], 90)),
                'exterior_p50': float(np.median(image[~mask])),
                'exterior_p90': float(np.percentile(image[~mask], 90))})

torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
rng = np.random.default_rng(20261011195)
synthetic = []
while len(synthetic) < 32:
    virtual = int(rng.integers(len(context['subjects'])))
    seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    with torch.inference_mode():
        sample = sample_one_shot_slide_artifacts_v4(context, [virtual], seed, side=256)
    if not bool(sample['eligible'][0]):
        continue
    image = sample['inputs'][0, 0].cpu().numpy()
    mask = (sample['visible'][0] > .05).cpu().numpy()
    synthetic.append({'physical_section_id': sample['provenance'][0]['physical_section_id'],
        'mode': sample['provenance'][0]['mode'],
        'exposure': sample['provenance'][0]['one_shot_slide_artifacts_v4']['exposure'],
        'whole_p99': float(np.percentile(image, 99)),
        'tissue_p50': float(np.median(image[mask])),
        'tissue_p90': float(np.percentile(image[mask], 90)),
        'exterior_p50': float(np.median(image[~mask])),
        'exterior_p90': float(np.percentile(image[~mask], 90))})

synthetic_v7 = []
while len(synthetic_v7) < 32:
    virtual = int(rng.integers(len(context['subjects'])))
    seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    with torch.inference_mode():
        sample = sample_one_shot_slide_artifacts_v7(context, [virtual], seed, side=256)
    if not bool(sample['eligible'][0]):
        continue
    image = sample['inputs'][0, 0].cpu().numpy()
    mask = (sample['visible'][0] > .05).cpu().numpy()
    synthetic_v7.append({'physical_section_id': sample['provenance'][0]['physical_section_id'],
        'mode': sample['provenance'][0]['mode'],
        'target_p99': sample['provenance'][0]['one_shot_slide_artifacts_v7'][
            'target_whole_image_p99_before_background'],
        'whole_p99': float(np.percentile(image, 99)),
        'tissue_p50': float(np.median(image[mask])),
        'tissue_p90': float(np.percentile(image[mask], 90)),
        'exterior_p50': float(np.median(image[~mask])),
        'exterior_p90': float(np.percentile(image[~mask], 90))})

summary = {'real_n': len(real), 'synthetic_v4_n': len(synthetic),
    'synthetic_v7_n': len(synthetic_v7),
    'real_animals': sorted({row['animal'] for row in real})}
for key in ('whole_p99', 'tissue_p50', 'tissue_p90', 'exterior_p50', 'exterior_p90'):
    summary[key] = {'real_p10_p50_p90': np.percentile(
        [row[key] for row in real], [10, 50, 90]).round(4).tolist(),
        'synthetic_v4_p10_p50_p90': np.percentile(
            [row[key] for row in synthetic], [10, 50, 90]).round(4).tolist(),
        'synthetic_v7_p10_p50_p90': np.percentile(
            [row[key] for row in synthetic_v7], [10, 50, 90]).round(4).tolist()}
out = root / 'reports/current_input_contrast_195_v7.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'summary': summary, 'real': real,
    'synthetic_v4': synthetic, 'synthetic_v7': synthetic_v7}, indent=2))
print(json.dumps(summary, indent=2))
