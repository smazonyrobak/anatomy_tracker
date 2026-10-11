"""Development-only weak-affine guard for the completed 158 comparison."""

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

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_156 import JointAnatomyModel156


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


train = root / 'runs/joint_feedback_158_train151_001'
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
out = root / 'runs/joint_feedback_158_weak_real_guard_001'
assert out.drive.upper() == 'I:' and not out.exists()
done = json.loads((train / 'completed.json').read_text())
assert done['calibrated'] is False and done['final_animals_used'] is False

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert coronal_done['development_images'] == 64 and coronal_done['development_donors'] == 6
for name in ('images.npy', 'geometry.npz', 'records.jsonl'):
    assert sha(coronal / name) == coronal_done['output_sha256'][name]
coronal_records = [record for line in (coronal / 'records.jsonl').open()
                   if (record := json.loads(line))['training_split'] == 'development']
assert len(coronal_records) == 64 and len({r['animal_id'] for r in coronal_records}) == 6
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()

sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert sagittal_summary['roles']['weak_dev2_source_available'] == {'donors': 8, 'sections': 158}
for name, digest in sagittal_summary['output_sha256'].items():
    assert sha(sagittal / name) == digest
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
assert len(sagittal_records) == 158 and len({r['donor_id'] for r in sagittal_records}) == 8
assert all(r['split'] == 'weak_dev2_source_available' for r in sagittal_records)
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = JointAnatomyModel156().cuda().eval().requires_grad_(False)
pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                       [127.5, 127.5]], device='cuda')
chart = pixels / 256
reflection_options = torch.arange(2, device='cuda')[None, None]
checkpoints = [('baseline', 'feedback', 0)] + [
    (arm, arm, step) for arm in ('feedback', 'detached_control') for step in (500, 1000)]
config = {'training_completion_sha256': sha(train / 'completed.json'),
          'coronal_completion_sha256': sha(coronal / 'completed.json'),
          'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
          'evaluator_sha256': sha(__file__),
          'checkpoint_sha256': {},
          'scope': 'donor-disjoint development images with inherited weak Allen affines; not expert anatomical truth',
          'metric': 'five fixed full-canvas CCF points and antipodal plane normal; direct prior-selected branch',
          'calibrated': False, 'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for label, arm, step in checkpoints:
        path = train / arm / f'joint_feedback_step_{step:05d}.pt'
        assert sha(path) == done['arms'][arm]['checkpoint_sha256'][str(step)]
        config['checkpoint_sha256'][label if label == 'baseline' else f'{arm}/{step}'] = sha(path)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        del state
        for domain, records in (('coronal', coronal_records), ('sagittal', sagittal_records)):
            for record in records:
                row_index = record['array_row_index']
                if domain == 'coronal':
                    native = np.concatenate((np.asarray(coronal_images[row_index], dtype=np.float32),
                                             np.zeros((4, 192, 192), dtype=np.float32)))[None]
                    image = F.interpolate(torch.from_numpy(native).cuda(), (256, 256),
                                          mode='bilinear', align_corners=False)
                    affine = torch.as_tensor(coronal_affines[row_index], device='cuda', dtype=torch.float32)
                    reference = (affine[:, 2] + (192 / 256) * pixels[:, :1] * affine[:, 0]
                                 + (192 / 256) * pixels[:, 1:] * affine[:, 1])
                    donor = record['animal_id']
                else:
                    native = np.concatenate((np.asarray(sagittal_images[row_index], dtype=np.float32),
                                             np.zeros((4, 256, 256), dtype=np.float32)))[None]
                    image = torch.from_numpy(native).cuda()
                    affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                                             device='cuda', dtype=torch.float32)
                    reference = (affine[:, 2] + pixels[:, :1] * affine[:, 0]
                                 + pixels[:, 1:] * affine[:, 1])
                    donor = record['donor_id']
                prediction = model.pose.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                proposed_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                proposed_flags = reflection_options.expand(1, model.pose.modes, 2)
                proposed = rigid_points_090(proposed_states, proposed_flags, chart[None, None])
                error = (proposed[0] - reference[None, None]).norm(dim=-1).mean(-1).flatten() / 1000
                order = prior.argsort(descending=True)
                weak_normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
                normals = full_frame_state_to_components(prediction['state'])[1][0, :, :, 2]
                normal_error = torch.rad2deg((normals @ weak_normal).abs().clamp(0, 1).acos())
                row = {'checkpoint': label if label == 'baseline' else f'{arm}/{step}',
                       'domain': domain, 'donor_id': donor,
                       'specimen_id': record['specimen_id'],
                       'experiment_id': record['experiment_id'],
                       'section_id': record['section_id'],
                       'array_row_index': row_index,
                       'top1_five_point_weak_mm': float(error[order[0]]),
                       'top16_best_five_point_weak_mm': float(error[order[:16]].min()),
                       'top1_normal_weak_deg': float(normal_error[order[0] // 2]),
                       'label_role': 'inherited weak Allen affine, not expert pose'}
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps({'checkpoint': label, 'domain': domain,
                              'sections_done': len(records)}), flush=True)

summary = {}
for label, arm, step in checkpoints:
    key = label if label == 'baseline' else f'{arm}/{step}'
    for domain in ('coronal', 'sagittal'):
        group = [r for r in rows if r['checkpoint'] == key and r['domain'] == domain]
        donors = sorted({r['donor_id'] for r in group})
        summary[f'{key}/{domain}'] = {'sections': len(group), 'donors': len(donors)}
        for metric in ('top1_five_point_weak_mm', 'top16_best_five_point_weak_mm',
                       'top1_normal_weak_deg'):
            summary[f'{key}/{domain}'][metric] = float(np.mean([
                np.mean([r[metric] for r in group if r['donor_id'] == donor])
                for donor in donors]))
(out / 'config.json').write_text(json.dumps(config, indent=2))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'weak_real_guard_complete', 'summary': summary}), flush=True)
