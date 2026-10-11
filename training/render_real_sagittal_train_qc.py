"""Six acquired sagittal-ish TRAIN donor images for synthetic-domain visual QC."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['MPLCONFIGDIR'] = str(root / 'cache/matplotlib')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

source = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
summary = json.loads((source / 'summary.json').read_text())
assert summary['roles'] == {'train': {'donors': 40, 'sections': 653}}
with (source / 'geometry.jsonl').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == summary['output_sha256']['geometry.jsonl']
with (source / 'model_input.npy').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == summary['output_sha256']['model_input.npy']
records = [json.loads(line) for line in (source / 'geometry.jsonl').open()]
images = np.load(source / 'model_input.npy', mmap_mode='r')
donors = {}
for index, record in enumerate(records):
    donors.setdefault(record['donor_id'], []).append(index)
assert len(donors) == 40 and len(records) == len(images) == 653
chosen = np.random.default_rng(20261011224).choice(sorted(donors), 6, replace=False).tolist()
figure, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
rows = []
for axis, donor in zip(axes.flat, chosen):
    index = donors[donor][len(donors[donor]) // 2]
    image = np.asarray(images[index, 0])
    record = records[index]
    axis.imshow(image, cmap='gray', vmin=0, vmax=1)
    axis.set_title(f'donor {donor} · section {record["section_id"]}', fontsize=9)
    axis.axis('off')
    rows.append({'donor_id': donor, 'index': index,
        'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
        'section_id': record['section_id'], 'image_sha256': record['image_sha256'],
        'input_mean': float(image.mean()), 'input_p99': float(np.percentile(image, 99))})
out = root / 'previews/real_sagittal_train_six_qc.png'
out.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(out, dpi=135)
plt.close(figure)
out.with_suffix('.json').write_text(json.dumps({'selection':
    'six random distinct TRAIN donors, midpoint section per donor; no image-quality selection',
    'summary_sha256': hashlib.sha256((source / 'summary.json').read_bytes()).hexdigest(),
    'rows': rows}, indent=2))
print(json.dumps({'image': str(out), 'rows': rows}), flush=True)
