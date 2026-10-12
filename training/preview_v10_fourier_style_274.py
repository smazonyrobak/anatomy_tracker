"""Eight independent generator outputs beside their true atlas planes."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_slide_artifacts_v10 import sample_one_shot_slide_artifacts_v10
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090


OUT = ROOT / 'previews/v10_fourier_style_274_002'
assert Path(__file__).drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
style_banks = {}
for domain, directory, record_name, array_name, donor_key in (
    ('coronal', ROOT / 'data/joint_v7_allen_fullcanvas_192_001', 'records.jsonl', 'images.npy', 'animal_id'),
    ('sagittal', ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008',
     'geometry.jsonl', 'model_input.npy', 'donor_id')):
    records = [json.loads(line) for line in (directory / record_name).open()]
    if domain == 'coronal':
        records = [row for row in records if row['training_split'] == 'train']
    donors = sorted({row[donor_key] for row in records if
                     hashlib.sha256(f'{domain}|{row[donor_key]}'.encode()).digest()[0] % 2 == 0})[:4]
    selected = [next(row for row in records if row[donor_key] == donor) for donor in donors]
    array = np.load(directory / array_name, mmap_mode='r')
    images = torch.from_numpy(np.stack([np.asarray(array[row['array_row_index'], 0],
                                                  dtype=np.float32) for row in selected]))[:, None].cuda()
    if domain == 'sagittal':
        images = F.interpolate(images, (192, 192), mode='area')
    style_banks[domain] = {'images': images[:, 0],
        'records': [{'donor_id': row[donor_key], 'specimen_id': row['specimen_id'],
                     'experiment_id': row['experiment_id'], 'section_id': row['section_id'],
                     'image_sha256': row.get('source_image_sha256', row.get('image_sha256'))}
                    for row in selected]}
context = load_streaming_synthetic_v7_64(device='cuda')
indices = [base * VARIANTS for base in (0, 2, 8, 10, 18, 26, 34, 42, 50, 58, 61, 63)]
with torch.inference_mode():
    sample = sample_one_shot_slide_artifacts_v10(context, indices, 20261012274, style_banks)
    keep = torch.nonzero(sample['eligible']).flatten().tolist()[:8]
    assert len(keep) == 8
    rendered = render_atlas_planes_090(context['atlas'], sample['state'][keep, None],
                                       sample['reflection'][keep, None], sample['offsets'][keep],
                                       sample['weights'][keep], source_shape=(192, 192))[:, 0]
assert len({sample['provenance'][row]['physical_section_id'] for row in keep}) == 8
assert all(bool(torch.isfinite(sample['inputs'][row]).all()) for row in keep)
fig, axes = plt.subplots(8, 2, figsize=(8, 26))
for row in range(8):
    source_row = keep[row]
    provenance = sample['provenance'][source_row]
    style = provenance['one_shot_slide_artifacts_v10']
    axes[row, 0].imshow(rendered[row, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[row, 1].imshow(sample['inputs'][source_row, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[row, 0].set_title(f'Atlas plane | base {indices[source_row] // VARIANTS}')
    axes[row, 1].set_title(f"Independent synthetic input | {style['style_domain'] or 'original v9'}")
    for ax in axes[row]:
        ax.axis('off')
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=False)
fig.savefig(OUT / 'atlas_and_input.png', dpi=120)
plt.close(fig)
(OUT / 'provenance.json').write_text(json.dumps([sample['provenance'][row] for row in keep], indent=2))
(OUT / 'completed.json').write_text(json.dumps({
    'cases': 8, 'attempted': len(indices),
    'styled': sum(sample['provenance'][row]['one_shot_slide_artifacts_v10']['fourier_style_applied']
                  for row in keep),
    'image_sha256': hashlib.sha256((OUT / 'atlas_and_input.png').read_bytes()).hexdigest(),
    'provenance_sha256': hashlib.sha256((OUT / 'provenance.json').read_bytes()).hexdigest()}, indent=2))
print(OUT / 'atlas_and_input.png')
