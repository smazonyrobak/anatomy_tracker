"""Freeze one 12k-section TRAIN draw and weak-real stream for both 128 arms."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import torch

from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

source = Path(__file__).resolve().parent
run = root / 'runs/joint_in_path_correspondence_128_manifest'
seed, updates, side = 20261010128, 6000, 256
draw_seed = seed * 1000000
protocol = source.parent / 'docs/publication/JOINT_IN_PATH_CORRESPONDENCE_128_PROTOCOL_20261010.md'
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
coronal = load_reserved_real_train()
sag_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert sha(sagittal_dir / 'geometry.jsonl') == sag_summary['output_sha256']['geometry.jsonl']
sag_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
assert all(row['split'] == 'train' for row in sag_records)
sag_by_donor = {}
for index, row in enumerate(sag_records):
    sag_by_donor.setdefault(row['donor_id'], []).append(index)
sag_donors = sorted(sag_by_donor)
rng = torch.Generator().manual_seed(seed)
config = {'seed': seed, 'updates': updates, 'synthetic_presentations': 2 * updates,
    'side': side, 'draw_seed_start': draw_seed,
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'prepare_joint_in_path_correspondence_128_manifest.py',
        'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py',
        'arbitrary_plane_streaming_synthetic_v7.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
        'arbitrary_plane_reserved_real_stream_v8.py')},
    'synthetic_provenance': context['provenance'],
    'coronal_bindings': coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_geometry_sha256': sha(sagittal_dir / 'geometry.jsonl'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))
attempts = 0
with (run / 'draws.jsonl').open('w') as draws, (run / 'updates.jsonl').open('w') as manifest:
    for step in range(1, updates + 1):
        synthetic = []
        for slot in range(2):
            while True:
                subject_index = int(torch.randint(len(context['subjects']), (1,), generator=rng))
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v3(context,
                        [subject_index], draw_seed, side=side)
                used = bool(sample['eligible'][0])
                record = {'update': step, 'slot': slot, 'draw_seed': draw_seed,
                    'subject_index': subject_index, 'attempt': attempts + 1,
                    'used': used, **sample['provenance'][0]}
                assert record['split'] == 'train'
                draws.write(json.dumps(record) + '\n')
                attempts += 1
                draw_seed += 1
                if used:
                    synthetic.append({**record,
                        'site_seed': int(torch.randint(2**31, (1,), generator=rng)),
                        'jitter_seed': int(torch.randint(2**31, (1,), generator=rng)),
                        'ce_jitter_seed': int(torch.randint(2**31, (1,), generator=rng))})
                    break
        coronal_donor = int(torch.randint(len(coronal['donors']), (1,), generator=rng))
        coronal_section = int(torch.randint(
            len(coronal['donors'][coronal_donor]['state']), (1,), generator=rng))
        sag_donor = sag_donors[int(torch.randint(len(sag_donors), (1,), generator=rng))]
        sag_rows = sag_by_donor[sag_donor]
        sag_index = sag_rows[int(torch.randint(len(sag_rows), (1,), generator=rng))]
        manifest.write(json.dumps({'update': step, 'synthetic': synthetic,
            'coronal_donor': coronal_donor, 'coronal_section': coronal_section,
            'coronal_identity': coronal['donors'][coronal_donor]['identities'][coronal_section],
            'sagittal_index': sag_index, 'sagittal_identity': {key: sag_records[sag_index][key]
                for key in ('donor_id', 'specimen_id', 'experiment_id', 'section_id')}}) + '\n')
        if step % 1000 == 0:
            draws.flush()
            manifest.flush()
            print(json.dumps({'event': 'manifest_milestone', 'update': step,
                'accepted': 2 * step, 'attempts': attempts}), flush=True)
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'accepted_synthetic': 2 * updates, 'synthetic_draw_attempts': attempts,
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'updates_sha256': sha(run / 'updates.jsonl'),
    'source_sha256': config['source_sha256'],
    'protocol_sha256': config['protocol_sha256'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False,
    'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'manifest_complete', 'updates': updates,
    'accepted': 2 * updates, 'attempts': attempts}), flush=True)
