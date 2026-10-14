"""Read-only TRAIN framing check; Lupori masks are near-coronal, not oblique truth."""
import copy
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

os.environ['TEMP'] = os.environ['TMP'] = 'I:/AnatomyTracker/tmp'
os.environ['TORCH_HOME'] = 'I:/AnatomyTracker/cache/torch'
sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image

from training.arbitrary_plane_fov_balance_v10 import FOV_SCALE, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import quicknii_to_allen_vectors
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64


def shape(mask):
    y, x = np.nonzero(mask)
    edge = (x < 2) | (x >= 190) | (y < 2) | (y >= 190)
    contacts = sum((np.any(x < 2), np.any(x >= 190), np.any(y < 2), np.any(y >= 190)))
    return np.array([len(x) / 192**2, x.mean() / 192, y.mean() / 192,
                     (x.max() - x.min() + 1) / 192, (y.max() - y.min() + 1) / 192,
                     edge.mean(), contacts], dtype=float)


root = Path('I:/AnatomyTracker')
run = root / 'runs/first_choice_exposure_309_001'
panel = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)['v10']
row = next(json.loads(line) for line in (run / 'single/readouts.jsonl').open(encoding='utf8')
           if json.loads(line)['step'] == 20000)
result = {r['physical_section_id']: r for r in row['rows'] if r['cohort'] == 'v10'}

real, animals, axes = [], [], []
with ZipFile(root / 'data/lupori_joint_185/rawData.zip') as archive:
    for animal in ('CC2B', 'CC4B', 'CC4A'):
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            archive.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        registration = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))
        for item in registration['slices']:
            section = Path(item['filename']).name.removesuffix('-thumb.png')
            if section not in valid:
                continue
            mask = np.asarray(Image.open(BytesIO(archive.read(
                f'{animal}/masks/{section}-mask.png'))).convert('L').resize(
                    (192, 192), Image.Resampling.NEAREST)) > 127
            ouv = torch.tensor(item['anchoring'], dtype=torch.float32).reshape(3, 3)
            normal = torch.cross(quicknii_to_allen_vectors(ouv[1]),
                                 quicknii_to_allen_vectors(ouv[2]), dim=0)
            real.append(shape(mask))
            animals.append(animal)
            axes.append(int(normal.abs().argmax()))
real = np.stack(real)

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
v9 = load_streaming_synthetic_v7_64('cuda')
v10 = {**v9, 'subjects': copy.deepcopy(v9['subjects']),
       'nuisance': v9['nuisance'].copy(), 'provenance': copy.deepcopy(v9['provenance'])}
for subject in v10['subjects']:
    bounds = np.asarray(subject['bounds_um'])
    subject['bounds_um'] = (bounds.mean(0) + FOV_SCALE * (bounds - bounds.mean(0))).tolist()
v10['nuisance'][:, 2:4] *= FOV_SCALE
v10['provenance']['fov_balance_v10'] = {'fov_scale': FOV_SCALE}

synthetic = []
with torch.inference_mode():
    for case in panel:
        meta = case['meta']
        sample = sample_fov_balance_v10(v10, [meta['base'] * 64 + meta['variant']],
                                        meta['seed'], 192)
        assert torch.equal(sample['inputs'][0].cpu(), case['inputs'])
        assert sample['provenance'][0]['physical_section_id'] == meta['physical_section_id']
        normal = full_frame_state_to_components(case['state'][None])[1][0, :, 2]
        measured = result[meta['physical_section_id']]
        synthetic.append({'shape': shape((sample['visible'][0] > .05).cpu().numpy()),
                          'axis': int(normal.abs().argmax()),
                          'obliquity': meta['obliquity_deg'],
                          'framing': meta['framing'],
                          'regret': measured['first_rigid_site_gap_mm'] -
                                    measured['top16_oracle_gap_mm']})

axes = np.asarray(axes)
print('TRAIN Lupori counts by normal family AP/DV/ML:', np.bincount(axes, minlength=3),
      'animals:', {animal: animals.count(animal) for animal in sorted(set(animals))})
names = ('tissue_fraction', 'centroid_x', 'centroid_y', 'bbox_width', 'bbox_height',
         'border_tissue_fraction', 'border_contact_sides')
for name, values in [('Lupori TRAIN AP', real[axes == 0]), *[(f'v10 {axis}', np.stack(
        [s['shape'] for s in synthetic if s['axis'] == index]))
        for index, axis in enumerate(('AP', 'DV', 'ML'))]]:
    print(name, 'n=', len(values), 'median=',
          dict(zip(names, np.round(np.median(values, axis=0), 3))), 'q05/q95=',
          {key: np.round(np.quantile(values[:, i], [.05, .95]), 3).tolist()
           for i, key in enumerate(names)})

reference = real[axes == 0][:, 1:5]
centre = reference.mean(0)
inverse = np.linalg.pinv(np.cov(reference, rowvar=False))
reference_distance = np.einsum('ni,ij,nj->n', reference - centre, inverse, reference - centre)
limit = np.quantile(reference_distance, .95)
print('AP crop-envelope limit (TRAIN Lupori 95th percentile Mahalanobis D2):',
      round(float(limit), 3))
for maximum_angle in (90, 30):
    selected = [s for s in synthetic if s['axis'] == 0 and s['obliquity'] <= maximum_angle]
    x = np.stack([s['shape'] for s in selected])
    difference = x[:, 1:5] - centre
    outside = np.einsum('ni,ij,nj->n', difference, inverse, difference) > limit
    visibility_bin = np.digitize(x[:, 0], [.15, .30, .50])
    regret = np.array([s['regret'] for s in selected])
    print('v10 AP obliquity<=', maximum_angle, 'n=', len(selected),
          'outside=', int(outside.sum()), 'mean regret outside/inside=',
          [round(float(regret[m].mean()), 3) if m.any() else None
           for m in (outside, ~outside)])
    for b, label in enumerate(('<15%', '15-30%', '30-50%', '>=50%')):
        print('  visibility', label, 'outside/inside n=',
              [int(np.sum((visibility_bin == b) & m)) for m in (outside, ~outside)],
              'regret=', [round(float(regret[(visibility_bin == b) & m].mean()), 3)
                         if np.any((visibility_bin == b) & m) else None
                         for m in (outside, ~outside)])
