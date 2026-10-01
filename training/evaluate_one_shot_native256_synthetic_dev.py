"""Frozen one-shot fitter readout on native-256 held-out synthetic deformation plans."""
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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

PANEL = ROOT / 'data/one_shot_native256_synthetic_dev_001'
TRAIN = ROOT / 'runs/one_shot_atlas_conditioned_warp_matched_001'
OUT = ROOT / 'runs/one_shot_atlas_conditioned_warp_native256_dev_001'
PATHS = {'parent': ROOT / 'runs/one_shot_joint_mixed_real_003/joint_step_20000.pt',
         'image_2000': TRAIN / 'image_only/warp_step_02000.pt',
         'image_4000': TRAIN / 'image_only/warp_step_04000.pt',
         'atlas_2000': TRAIN / 'atlas_conditioned/warp_step_02000.pt',
         'atlas_4000': TRAIN / 'atlas_conditioned/warp_step_04000.pt'}
SIDE = 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


completed = json.loads((PANEL / 'completed.json').read_text())
protocol = json.loads((PANEL / 'protocol.json').read_text())
assert completed['physical_sections'] == 128 and completed['animals'] == 4
assert protocol['side'] == SIDE and sha(PANEL / 'protocol.json') == completed['protocol_sha256']
assert sha(PANEL / 'records.jsonl') == completed['records_sha256']
assert sha(Path(__file__).with_name('arbitrary_plane_one_shot_stream.py')) == \
    protocol['source_sha256']['arbitrary_plane_one_shot_stream.py']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open()]
assert len(records) == 128 and len({row['panel_physical_section_id'] for row in records}) == 128
assert len({row['animal_id'] for row in records}) == 4
for record in records:
    assert sha(PANEL / record['file']) == record['sha256']
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
models, checkpoint_hashes = {}, {}
for name, path in PATHS.items():
    checkpoint_hashes[name] = sha(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == (20000 if name == 'parent' else int(name[-4:]))
    model = OneShotJointSliceModel(atlas_conditioning=name.startswith('atlas_')).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    models[name] = model
    del checkpoint
OUT.mkdir(parents=True, exist_ok=False)
source_hashes = {name: sha(Path(__file__).with_name(name)) for name in
                 ('evaluate_one_shot_native256_synthetic_dev.py', 'arbitrary_plane_one_shot_model.py',
                  'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
                  'arbitrary_plane_ribbon_v6.py')}
rows = []
with torch.inference_mode():
    for record in records:
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            reflection = torch.tensor([[int(arrays['reflection'])]], device='cuda')
            eligible = bool(arrays['eligible'])
        parent_prediction = models['parent'].predict(image)
        log_reflection = torch.stack((F.logsigmoid(-parent_prediction['reflection_logit']),
                                      F.logsigmoid(parent_prediction['reflection_logit'])), -1)
        selected = int((parent_prediction['log_mass'][..., None] + log_reflection).flatten(1).argmax(-1)[0])
        selected_mode = torch.tensor([[selected // 2]], device='cuda')
        selected_reflection = torch.tensor([[selected % 2]], device='cuda')
        row = {'panel_physical_section_id': record['panel_physical_section_id'],
               **{key: record[key] for key in ('animal_id', 'subject_id', 'specimen_id', 'experiment_id', 'section_id')},
               'appearance_mode': record['appearance_mode'],
               'warp_strength': record['provenance']['one_shot']['warp_strength'],
               'damage_events': record['provenance']['one_shot']['events'],
               'input_sha256': record['sha256'], 'eligible': eligible,
               'valid_pixels': int(valid.sum()), 'selected_mode': selected // 2,
               'selected_reflection': selected % 2, 'true_reflection': int(reflection[0, 0])}
        for name, model in models.items():
            prediction = parent_prediction if name == 'parent' else model.predict(image)
            state = prediction['state'].clone()
            state[:, 0] = truth
            mapped_true = model.map({**prediction, 'state': state}, offsets,
                                    torch.zeros((1, 1), device='cuda', dtype=torch.long),
                                    reflection, (SIDE, SIDE), atlas, weights)
            mapped_pred = model.map({**prediction, 'state': parent_prediction['state']}, offsets,
                                    selected_mode, selected_reflection, (SIDE, SIDE), atlas, weights)
            for label, mapped in (('true', mapped_true), ('predicted', mapped_pred)):
                surface = mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0]
                if bool(valid.any()):
                    error = (surface - target).norm(dim=-1)[valid]
                    row[f'{name}_{label}_mean_um'] = float(error.mean())
                    row[f'{name}_{label}_median_um'] = float(error.median())
                    if name == 'parent':
                        frame = full_frame_state_to_components(mapped['state'][:, 0])[1]
                        field = mapped['local_displacement_um'][:, 0]
                        rigid = surface - torch.einsum('bij,bjhw->bhwi', frame, field)
                        baseline = (rigid - target).norm(dim=-1)[valid]
                        row[f'zero_{label}_mean_um'] = float(baseline.mean())
                        row[f'zero_{label}_median_um'] = float(baseline.median())
                    row[f'{name}_{label}_gain_um'] = row[f'zero_{label}_mean_um'] - row[f'{name}_{label}_mean_um']
                else:
                    row[f'{name}_{label}_mean_um'] = None
                    row[f'{name}_{label}_median_um'] = None
                    row[f'{name}_{label}_gain_um'] = None
                    if name == 'parent':
                        row[f'zero_{label}_mean_um'] = None
                        row[f'zero_{label}_median_um'] = None
        rows.append(row)
        print(json.dumps({'evaluated': len(rows), 'eligible': eligible,
                          'animal_id': record['animal_id']}), flush=True)
with (OUT / 'rows.jsonl').open('w') as stream:
    stream.writelines(json.dumps(row) + '\n' for row in rows)
metrics = [f'{name}_{pose}_mean_um' for name in ('zero', *PATHS) for pose in ('true', 'predicted')]
metrics += [f'{name}_{pose}_gain_um' for name in PATHS for pose in ('true', 'predicted')]
groups = {'eligible': [row for row in rows if row['eligible']], 'all': rows,
          'censored': [row for row in rows if not row['eligible']]}
for mode in ('raw', 'exact_black', 'imperfect_brush'):
    groups[f'eligible_{mode}'] = [row for row in rows if row['eligible'] and row['appearance_mode'] == mode]
summary = {'scope': 'native256 synthetic DEVELOPMENT deformation plans; one atlas, not biological animals or final validation',
           'panel_protocol_sha256': completed['protocol_sha256'], 'panel_records_sha256': completed['records_sha256'],
           'checkpoint_sha256': checkpoint_hashes, 'evaluator_source_sha256': source_hashes,
           'all_rows': len(rows), 'groups': {}}
for group, items in groups.items():
    scored = [row for row in items if row['zero_true_mean_um'] is not None]
    animals = sorted({row['animal_id'] for row in scored})
    summary['groups'][group] = {'raw_sections': len(items), 'scored_sections': len(scored),
        'animals': len(animals),
        'draw_mean': {key: float(np.mean([row[key] for row in scored])) if scored else None for key in metrics},
        'animal_equal_mean': {key: float(np.mean([np.mean([row[key] for row in scored
                                                       if row['animal_id'] == animal]) for animal in animals]))
                              if animals else None for key in metrics}}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'completed.json').write_text(json.dumps({'rows': len(rows), 'eligible': sum(row['eligible'] for row in rows),
    'censored': sum(not row['eligible'] for row in rows),
    'panel_protocol_sha256': completed['protocol_sha256'], 'panel_records_sha256': completed['records_sha256'],
    'checkpoint_sha256': checkpoint_hashes, 'source_sha256': source_hashes,
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'complete_rows': len(rows), 'eligible': summary['groups']['eligible']}), flush=True)
