"""Matched head-only training of shared versus untied 132 anchor corrections."""

import hashlib
import json
import math
import os
import sys
import time
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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_reserved_real_stream_v8 import (
    load_reserved_real_train, sample_reserved_real_train)
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.shared_anchor_pose_138 import ARMS, AnchorResidual138, UPDATE_LIMITS, corrected_states


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/SHARED_ANCHOR_POSE_138_PROTOCOL_20261010.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/shared_anchor_pose_138'
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
seed, side, updates, sites = 20261010138, 256, 4000, 128
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(parent_done.get(key, False) for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
coronal = load_reserved_real_train()
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
assert coronal['bindings'] == parent_config['coronal_bindings']
assert len(context['bases']) == 64 and len(context['subjects']) == 4096

sagittal_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert sha(sagittal_dir / 'summary.json') == parent_config['sagittal_summary_sha256']
assert all(sha(sagittal_dir / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
assert all(row['split'] == 'train' for row in sagittal_records)
sagittal_images = np.load(sagittal_dir / 'model_input.npy', mmap_mode='r')
sagittal_donors = {}
for index, record in enumerate(sagittal_records):
    sagittal_donors.setdefault(record['donor_id'], []).append(index)
assert len(sagittal_records) == 653 and len(sagittal_donors) == 40
sagittal_donor_ids = sorted(sagittal_donors)
affine = torch.tensor(np.asarray([row['model_pixel_to_ccf_ref9_ap_dv_ml_um']
    for row in sagittal_records]), dtype=torch.float64)
ouv = torch.stack((affine[:, :, 2], side * affine[:, :, 0], side * affine[:, :, 1]), 1)
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
normal = full_frame_state_to_components(sagittal_states)[1][..., :, 2]
sagittal_reflection = normal.gather(1, normal.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sagittal_reflection, 0] += (side - 1) / side * ouv[sagittal_reflection, 1]
ouv[sagittal_reflection, 1] *= -1
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

heads = {}
for arm in ARMS:
    torch.manual_seed(seed + (arm == 'untied_visual'))
    heads[arm] = AnchorResidual138(arm).cuda().train()
heads['shared_source_zero'].load_state_dict(heads['shared_visual'].state_dict())
optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    for arm, head in heads.items()}
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
source_files = ('train_shared_anchor_pose_138.py', 'shared_anchor_pose_138.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'updates': updates, 'side': side, 'valid_sites': sites,
    'arms': ARMS, 'synthetic_per_update': {'v4': 2, 'v3': 1},
    'real_weak_per_update': {'coronal': 1, 'sagittal': 1},
    'checkpoints': (0, updates), 'update_limits': UPDATE_LIMITS,
    'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_config_sha256': sha(parent_dir / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'], 'coronal_bindings': coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_output_sha256': sagittal_summary['output_sha256'],
    'optimised': 'only three separate 138 residual heads; parent 132 entirely frozen',
    'loss': 'four nearest-normal anchors, correct reflection; synthetic 0.75 tissue + 0.25 five-point + 132 normal penalty; 0.5 weak-real five-point',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
for arm in ARMS:
    (run / arm).mkdir()
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    for arm, head in heads.items():
        path = run / arm / f'head_step_{step:05d}.pt'
        temp = path.with_suffix('.tmp')
        torch.save({'step': step, 'arm': arm, 'head': head.state_dict(),
            'optimizer': optimizers[arm].state_dict(), 'config': config,
            'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all()}, temp)
        os.replace(temp, path)


save(0)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
started = time.perf_counter()
draw_count = 0
accepted_ids = set()
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        samples = []
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            while True:
                virtual = int(rng.integers(len(context['subjects'])))
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                eligible = bool(sample['eligible'][0])
                record = sample['provenance'][0]
                draw_count += 1
                draws.write(json.dumps({'kind': 'synthetic_train', 'update': step,
                    'slot': slot, 'appearance_version': appearance,
                    'draw_attempt': draw_count, 'used': eligible, **record},
                    allow_nan=False) + '\n')
                if eligible:
                    accepted_ids.add(record['physical_section_id'])
                    samples.append(sample)
                    break

        donor_index = int(rng.integers(len(coronal['donors'])))
        section_index = int(rng.integers(len(coronal['donors'][donor_index]['identities'])))
        real_coronal = sample_reserved_real_train(coronal, donor_index, [section_index])
        coronal_inputs = F.interpolate(real_coronal['inputs'], (side, side),
            mode='bilinear', align_corners=False)
        donor_id = sagittal_donor_ids[int(rng.integers(len(sagittal_donor_ids)))]
        sag_rows = sagittal_donors[donor_id]
        sag_index = sag_rows[int(rng.integers(len(sag_rows)))]
        sagittal_inputs = torch.zeros(1, 5, side, side, device='cuda')
        sagittal_inputs[:, :1] = torch.from_numpy(np.asarray(
            sagittal_images[sag_index:sag_index + 1]).copy()).to('cuda')
        coronal_identity = real_coronal['identities'][0]
        sagittal_identity = {key: sagittal_records[sag_index][key] for key in
            ('donor_id', 'specimen_id', 'experiment_id', 'section_id', 'image_sha256')}
        draws.write(json.dumps({'kind': 'coronal_weak_train', 'update': step,
            'slot': 3, 'used': True, **coronal_identity}, allow_nan=False) + '\n')
        draws.write(json.dumps({'kind': 'sagittal_weak_train', 'update': step,
            'slot': 4, 'used': True, **sagittal_identity}, allow_nan=False) + '\n')

        inputs = torch.cat([sample['inputs'] for sample in samples]
                           + [coronal_inputs, sagittal_inputs])
        target_state = torch.cat([sample['state'] for sample in samples])
        target_reflection = torch.cat([sample['reflection'] for sample in samples]).long()
        valid = torch.cat([sample['valid_mask'] for sample in samples])
        real_state = torch.cat((real_coronal['state'],
            sagittal_states[sag_index:sag_index + 1].to('cuda')))
        real_reflection = torch.cat((real_coronal['reflection'],
            sagittal_reflection[sag_index:sag_index + 1].long().to('cuda'))).long()
        with torch.no_grad():
            prediction = model.predict(inputs)
            true_normal = full_frame_state_to_components(
                torch.cat((target_state, real_state)))[1][..., :, 2]
            near = ((true_normal @ model.normal_anchor_frames[:, :, 2].T).abs()
                .topk(4, -1).indices + model.base_modes)
            pixel = torch.multinomial(valid.flatten(1).float(), sites, replacement=True)
            chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            target_tissue = rigid_points_090(target_state, target_reflection, chart)
            target_five = rigid_points_090(target_state, target_reflection, five)
            real_five = rigid_points_090(real_state, real_reflection, five)

        arm_rows = {}
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        for arm, head in heads.items():
            states = corrected_states(prediction, head, model)
            chosen = states.gather(1, near[..., None].expand(-1, -1, 12))
            synthetic = chosen[:3]
            reflected = target_reflection[:, None].expand(-1, 4)
            proposed_tissue = rigid_points_090(synthetic, reflected,
                chart[:, None])
            proposed_five = rigid_points_090(synthetic, reflected, five)
            tissue_mm = (proposed_tissue - target_tissue[:, None]).norm(dim=-1).mean(-1)
            five_mm = (proposed_five - target_five[:, None]).norm(dim=-1).mean(-1)
            proposed_normal = full_frame_state_to_components(synthetic)[1][..., :, 2]
            normal_penalty = 4000 * (1 - (proposed_normal * true_normal[:3, None])
                .sum(-1).abs().clamp_max(1))
            synthetic_loss = (.75 * tissue_mm + .25 * five_mm + normal_penalty).mean() / 1000
            real_proposed = rigid_points_090(chosen[3:],
                real_reflection[:, None].expand(-1, 4), five)
            real_loss = (real_proposed - real_five[:, None]).norm(dim=-1).mean() / 1000
            loss = synthetic_loss + .5 * real_loss
            optimizer = optimizers[arm]
            for group in optimizer.param_groups:
                group['lr'] = 2e-4 * decay
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
            optimizer.step()
            arm_rows[arm] = {'synthetic_loss': float(synthetic_loss.detach()),
                'real_weak_loss': float(real_loss.detach()), 'loss': float(loss.detach()),
                'gradient_norm': float(gradient)}
        row = {'update': step, 'synthetic_presentations': 3 * step,
            'v4_presentations': 2 * step, 'v3_presentations': step,
            'synthetic_section_ids': [sample['provenance'][0]['physical_section_id']
                for sample in samples], 'coronal_identity': coronal_identity,
            'sagittal_identity': sagittal_identity, 'arms': arm_rows,
            'seconds_since_process_start': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({'event': 'train_milestone', 'update': step,
                'losses': {arm: arm_rows[arm]['loss'] for arm in ARMS}}), flush=True)
    save(updates)

assert len(accepted_ids) == 3 * updates
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'synthetic_presentations': 3 * updates, 'v4_presentations': 2 * updates,
    'v3_presentations': updates, 'synthetic_draw_attempts': draw_count,
    'unique_synthetic_physical_sections': len(accepted_ids),
    'real_weak_coronal_presentations': updates,
    'real_weak_sagittal_presentations': updates,
    'parent_checkpoint_sha256': sha(parent), 'protocol_sha256': sha(protocol),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {arm: {str(step): sha(run / arm /
        f'head_step_{step:05d}.pt') for step in (0, updates)} for arm in ARMS},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
