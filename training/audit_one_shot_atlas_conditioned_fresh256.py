"""Fresh 256px fitter audit on training deformation bases, not independent animals."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

SIDE, WANTED, SEED = 256, 128, 20261005070000000
PARENT = ROOT / 'runs/one_shot_joint_mixed_real_003/joint_step_20000.pt'
TRAIN = ROOT / 'runs/one_shot_atlas_conditioned_warp_matched_001'
OUT = ROOT / 'runs/one_shot_atlas_conditioned_warp_fresh256_audit_001'
DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
PATHS = {'parent': PARENT,
         'image_2000': TRAIN / 'image_only/warp_step_02000.pt',
         'image_4000': TRAIN / 'image_only/warp_step_04000.pt',
         'atlas_2000': TRAIN / 'atlas_conditioned/warp_step_02000.pt',
         'atlas_4000': TRAIN / 'atlas_conditioned/warp_step_04000.pt'}
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert json.loads((TRAIN / 'image_only/completed.json').read_text())['updates'] == 4000
assert json.loads((TRAIN / 'atlas_conditioned/completed.json').read_text())['updates'] == 4000
context = load_streaming_synthetic_v7_64(device='cuda')
dev_animals = {json.loads(line)['animal_id'] for line in (DEV / 'records.jsonl').open()}
assert len(context['bases']) == 64 and len(dev_animals) == 4
assert not dev_animals & {base['lineage']['animal_id'] for base in context['bases']}
models, checkpoint_hashes = {}, {}
for name, path in PATHS.items():
    with path.open('rb') as stream:
        checkpoint_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == (20000 if name == 'parent' else int(name[-4:]))
    conditioned = name.startswith('atlas_')
    model = OneShotJointSliceModel(atlas_conditioning=conditioned).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    models[name] = model
    del checkpoint
source_hashes = {}
for name in ('audit_one_shot_atlas_conditioned_fresh256.py', 'arbitrary_plane_one_shot_model.py',
             'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7.py',
             'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_full_frame_primitives.py',
             'arbitrary_plane_geometry.py', 'arbitrary_plane_ribbon_v6.py'):
    with (Path(__file__).parent / name).open('rb') as stream:
        source_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
OUT.mkdir(parents=True, exist_ok=False)

subjects_rng = torch.Generator().manual_seed(SEED)
rows, attempts, attempt = [], [], 0
with torch.inference_mode():
    while len(rows) < WANTED:
        subjects = torch.randint(len(context['subjects']), (4,), generator=subjects_rng).tolist()
        batch = sample_one_shot_stream(context, subjects, SEED + attempt, side=SIDE)
        for slot, identity in enumerate(batch['provenance']):
            eligible = bool(batch['eligible'][slot])
            used = eligible and len(rows) < WANTED
            attempts.append({'attempt': attempt, 'slot': slot, 'eligible': eligible, 'used': used,
                             'physical_section_id': identity['physical_section_id']})
            if not used:
                continue
            image = batch['inputs'][slot:slot + 1]
            truth = batch['state'][slot:slot + 1]
            reflection = batch['reflection'][slot:slot + 1, None]
            target = batch['centre'][slot:slot + 1]
            valid = batch['valid_mask'][slot:slot + 1]
            offsets = batch['offsets'][slot:slot + 1]
            weights = batch['weights'][slot:slot + 1]
            parent_prediction = models['parent'].predict(image)
            log_reflection = torch.stack((F.logsigmoid(-parent_prediction['reflection_logit']),
                                          F.logsigmoid(parent_prediction['reflection_logit'])), -1)
            selected = int((parent_prediction['log_mass'][..., None] + log_reflection).flatten(1).argmax(-1)[0])
            selected_mode = torch.tensor([[selected // 2]], device='cuda')
            selected_reflection = torch.tensor([[selected % 2]], device='cuda')
            row = {'draw': len(rows), 'attempt': attempt, 'slot': slot, 'side': SIDE,
                   'physical_section_id': identity['physical_section_id'],
                   'animal_id': identity['base_lineage']['animal_id'],
                   'specimen_id': identity['base_lineage']['specimen_id'],
                   'experiment_id': identity['base_lineage']['experiment_id'],
                   'section_id': identity['physical_section_id'],
                   'virtual_subject_id': identity['virtual_subject_id'],
                   'appearance_mode': identity['mode'],
                   'warp_strength': identity['one_shot']['warp_strength'],
                   'damage_events': identity['one_shot']['events'],
                   'valid_pixels': int(valid.sum()), 'selected_mode': selected // 2,
                   'selected_reflection': selected % 2, 'true_reflection': int(reflection[0, 0]),
                   'provenance': identity}
            for name, model in models.items():
                prediction = parent_prediction if name == 'parent' else model.predict(image)
                state = prediction['state'].clone()
                state[:, 0] = truth
                mapped_true = model.map({**prediction, 'state': state}, offsets,
                                        torch.zeros((1, 1), device='cuda', dtype=torch.long),
                                        reflection, (SIDE, SIDE), context['atlas'], weights)
                mapped_pred = model.map({**prediction, 'state': parent_prediction['state']}, offsets,
                                        selected_mode, selected_reflection,
                                        (SIDE, SIDE), context['atlas'], weights)
                for label, mapped in (('true', mapped_true), ('predicted', mapped_pred)):
                    surface = mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0]
                    error = (surface - target).norm(dim=-1)[valid]
                    row[f'{name}_{label}_mean_um'] = float(error.mean())
                    row[f'{name}_{label}_median_um'] = float(error.median())
                    if name == 'parent':
                        field = mapped['local_displacement_um'][:, 0]
                        frame = full_frame_state_to_components(mapped['state'][:, 0])[1]
                        rigid = surface - torch.einsum('bij,bjhw->bhwi', frame, field)
                        baseline = (rigid - target).norm(dim=-1)[valid]
                        row[f'zero_{label}_mean_um'] = float(baseline.mean())
                        row[f'zero_{label}_median_um'] = float(baseline.median())
                    row[f'{name}_{label}_gain_um'] = row[f'zero_{label}_mean_um'] - row[f'{name}_{label}_mean_um']
            rows.append(row)
        attempt += 1
        if len(rows) % 32 == 0:
            print(json.dumps({'accepted': len(rows), 'attempt_batches': attempt}), flush=True)

assert len(rows) == WANTED and len({row['physical_section_id'] for row in rows}) == WANTED
with (OUT / 'rows.jsonl').open('w') as stream:
    stream.writelines(json.dumps(row) + '\n' for row in rows)
with (OUT / 'attempts.jsonl').open('w') as stream:
    stream.writelines(json.dumps(row) + '\n' for row in attempts)

metrics = [f'{name}_{pose}_mean_um' for name in ('zero', *PATHS) for pose in ('true', 'predicted')]
metrics += [f'{name}_{pose}_gain_um' for name in PATHS for pose in ('true', 'predicted')]
strength = np.array([row['warp_strength'] for row in rows])
edges = np.quantile(strength, [0, 1 / 3, 2 / 3, 1])
groups = {'all': rows}
for index in range(3):
    groups[f'warp_tertile_{index + 1}'] = [row for row in rows if
        edges[index] <= row['warp_strength'] and
        (row['warp_strength'] < edges[index + 1] if index < 2 else row['warp_strength'] <= edges[index + 1])]
for mode in ('raw', 'exact_black', 'imperfect_brush'):
    groups[f'appearance_{mode}'] = [row for row in rows if row['appearance_mode'] == mode]
summary = {'label': 'fresh current-generator 256px TRAIN-base diagnostic, not independent biology or final validation',
           'seed': SEED, 'wanted': WANTED, 'attempt_batches': attempt,
           'unique_training_deformation_animals': len({row['animal_id'] for row in rows}),
           'warp_strength_tertile_edges': edges.tolist(),
           'checkpoint_sha256': checkpoint_hashes, 'source_sha256': source_hashes,
           'synthetic_provenance': context['provenance'], 'groups': {}}
for group, subset in groups.items():
    animals = sorted({row['animal_id'] for row in subset})
    summary['groups'][group] = {'draws': len(subset), 'animals': len(animals),
        'draw_mean': {key: float(np.mean([row[key] for row in subset])) for key in metrics},
        'animal_equal_mean': {key: float(np.mean([np.mean([row[key] for row in subset
                                                       if row['animal_id'] == animal]) for animal in animals]))
                              for key in metrics}}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
output_hashes = {}
for name in ('rows.jsonl', 'attempts.jsonl', 'summary.json'):
    with (OUT / name).open('rb') as stream:
        output_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(OUT / 'completed.json').write_text(json.dumps({'rows': len(rows), 'attempt_rows': len(attempts),
    'checkpoint_sha256': checkpoint_hashes, 'source_sha256': source_hashes,
    'output_sha256': output_hashes, 'public_benchmark_used': False, 'calibrated': False}, indent=2))
print(json.dumps({'completed': len(rows), 'summary': summary['groups']['all']}), flush=True)
