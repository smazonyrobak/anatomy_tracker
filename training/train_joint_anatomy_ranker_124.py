"""Matched anatomy-intensity versus support-only ranker on frozen joint-122 maps."""

import copy
import hashlib
import json
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

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.atlas_anatomy_ranker_124 import (
    atlas_anatomy_rank_124, make_atlas_anatomy_ranker_124)
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match
from training.joint_pose_map_121 import joint_target_loss_121


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/JOINT_ANATOMY_RANKER_124_PROTOCOL_20261009.md'
parent_run = root / 'runs/joint_pose_map_122'
parent_path = parent_run / 'joint_step_02000.pt'
run = root / 'runs/joint_anatomy_ranker_124'
seed, presentations, side, map_side = 20261009124, 1500, 256, 96
checkpoints = (0, 500, 1000, 1500)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
parent_done = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_run / 'config.json') == parent_done['config_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_done['draws_sha256']
assert all(sha(source / name) == digest
           for name, digest in parent_config['source_sha256'].items())
prior_ids = {row['physical_section_id'] for row in map(
    json.loads, (parent_run / 'draws.jsonl').open()) if 'physical_section_id' in row}
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
atlas = context['atlas']
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['config'] == parent_config and not parent['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(parent['model'], strict=True)
del parent

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
full_ranker = make_atlas_anatomy_ranker_124().cuda().train()
support_ranker = copy.deepcopy(full_ranker).train()
full_optimizer = torch.optim.AdamW(full_ranker.parameters(), lr=3e-4, weight_decay=1e-4)
support_optimizer = torch.optim.AdamW(support_ranker.parameters(), lr=3e-4, weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
psf_offsets = torch.linspace(-31.25, 31.25, 9, device='cuda')[None]
psf_weights = torch.tensor([[1., 2., 2., 2., 2., 2., 2., 2., 1.]], device='cuda') / 16

source_names = ('train_joint_anatomy_ranker_124.py', 'atlas_anatomy_ranker_124.py',
    'joint_pose_map_121.py', 'global_plane_matcher_120.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py')
config = {'seed': seed, 'presentations': presentations, 'checkpoints': checkpoints,
    'side': side, 'map_side': map_side, 'candidate_count': 16,
    'beam': 'prior top-eight base + top-six anchor + two diverse anchor',
    'fixed_psf_um': 62.5, 'psf_samples': 9, 'ranker_depth_offsets_um': [-500, 0, 500],
    'ranker_local_xy_radius_cells': 2, 'ranker_side': 32,
    'hard_pair': 'best mapped<=1.5mm, wrong mapped>best+1mm, support Dice>=.85, area gap<=.08, common>=32',
    'optimizer': 'separate equal-initialized AdamW 3e-4; 1 selected hard pair per eligible presentation',
    'loss': 'softplus(0.2 - paired fitted-anatomy score margin)',
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'parent_config_sha256': sha(parent_run / 'config.json'),
    'parent_draws_sha256': sha(parent_run / 'draws.jsonl'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_names},
    'synthetic_provenance': context['provenance'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'step': step, 'full_ranker': full_ranker.state_dict(),
        'support_ranker': support_ranker.state_dict(),
        'full_optimizer': full_optimizer.state_dict(),
        'support_optimizer': support_optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'config': config, 'calibrated': False},
        run / f'ranker_step_{step:05d}.pt')


save(0)
attempts, scored = 0, 0
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as train_stream, (run / 'draws.jsonl').open('w') as draw_stream:
    for presentation in range(1, presentations + 1):
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            with torch.no_grad():
                sample = sample_one_shot_slide_artifacts_v3(context, virtual,
                    draw_seed, side=side)
            record = sample['provenance'][0]
            assert record['base_lineage']['split'] == 'train'
            assert record['physical_section_id'] not in prior_ids
            attempts += 1
            used = bool(sample['eligible'][0])
            draw_stream.write(json.dumps({'presentation': presentation,
                'draw_seed': draw_seed, 'draw_attempt': attempts, 'used': used,
                **record}, allow_nan=False) + '\n')
            draw_seed += 1
            if used:
                accepted = sample

        image = accepted['inputs']
        valid = accepted['valid_mask']
        with torch.no_grad():
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            for _ in range(2):
                chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
                similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                diversity = -similarity
                diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                anchor = diversity.argmax(-1)
                reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
                beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
            mode, reflection = beam // 2, beam % 2
            match = global_plane_match(model, prediction, mode, reflection,
                psf_offsets, psf_weights, atlas, (side, side), side=24)
            mapped = model.map({**prediction, 'state': match['state']}, psf_offsets,
                torch.arange(16, device='cuda')[None], reflection, (map_side, map_side),
                atlas, psf_weights, source_shape=(side, side))
            site_indices = torch.multinomial(valid.flatten(1).float(), 128, replacement=True)
            errors = joint_target_loss_121(mapped, valid, accepted['centre'],
                accepted['state'], accepted['reflection'].long(), site_indices,
                source_shape=(side, side))
            mapped_error = errors['map_loss'][0]
            rigid_error = errors['rigid_loss'][0]
            best = int(mapped_error.argmin())

        full = atlas_anatomy_rank_124(full_ranker, prediction, image,
            match['state'], reflection, atlas,
            mapped_coordinates=mapped['coordinates'],
            local_displacement_um=mapped['local_displacement_um'])
        support = atlas_anatomy_rank_124(support_ranker, prediction, image,
            match['state'], reflection, atlas,
            mapped_coordinates=mapped['coordinates'],
            local_displacement_um=mapped['local_displacement_um'], support_only=True)
        with torch.no_grad():
            mask = full['centre_support'][0]
            overlap = (mask & mask[best]).sum((-2, -1)).float()
            dice = 2 * overlap / (mask.sum((-2, -1)) + mask[best].sum()).clamp_min(1)
            area_gap = (mask.float().mean((-2, -1)) - mask[best].float().mean()).abs()
            eligible = ((mapped_error > mapped_error[best] + 1.)
                & (dice >= .85) & (area_gap <= .08)
                & full['pair_valid'][0, best])
            wrong = int(mapped_error.masked_fill(~eligible, torch.inf).argmin()) \
                if float(mapped_error[best]) <= 1.5 and bool(eligible.any()) else None
        full_loss = support_loss = None
        full_margin = support_margin = None
        if wrong is not None:
            full_margin = full['pair_margin'][0, best, wrong]
            support_margin = support['pair_margin'][0, best, wrong]
            full_loss = F.softplus(.2 - full_margin)
            support_loss = F.softplus(.2 - support_margin)
            full_optimizer.zero_grad(set_to_none=True)
            support_optimizer.zero_grad(set_to_none=True)
            full_loss.backward()
            support_loss.backward()
            torch.nn.utils.clip_grad_norm_(full_ranker.parameters(), 5., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(support_ranker.parameters(), 5., error_if_nonfinite=True)
            full_optimizer.step()
            support_optimizer.step()
            scored += 1

        row = {'presentation': presentation, 'physical_section_id': record['physical_section_id'],
            'base_subject_id': record['base_lineage']['subject_id'],
            'draw_seed': draw_seed - 1, 'best_slot': best,
            'best_mapped_mm': float(mapped_error[best]),
            'best_rigid_mm': float(rigid_error[best]),
            'wrong_slot': wrong, 'wrong_mapped_mm': float(mapped_error[wrong])
                if wrong is not None else None,
            'support_dice': float(dice[wrong]) if wrong is not None else None,
            'support_area_gap': float(area_gap[wrong]) if wrong is not None else None,
            'common_pixels': int(full['pair_common_count'][0, best, wrong])
                if wrong is not None else None,
            'full_margin': float(full_margin.detach()) if wrong is not None else None,
            'support_margin': float(support_margin.detach()) if wrong is not None else None,
            'full_loss': float(full_loss.detach()) if wrong is not None else None,
            'support_loss': float(support_loss.detach()) if wrong is not None else None,
            'scored_total': scored, 'seconds': time.perf_counter() - started}
        train_stream.write(json.dumps(row, allow_nan=False) + '\n')
        if presentation == 1 or presentation % 500 == 0:
            train_stream.flush()
            draw_stream.flush()
            print(json.dumps({'presentation': presentation, 'scored': scored,
                'eligible_fraction': scored / presentation,
                'full_loss': row['full_loss'], 'support_loss': row['support_loss'],
                'seconds': row['seconds']}), flush=True)
        if presentation in checkpoints:
            save(presentation)
    train_stream.flush()
    draw_stream.flush()

(run / 'completed.json').write_text(json.dumps({
    'presentations': presentations, 'synthetic_draw_attempts': attempts,
    'scored_pairs': scored, 'parent_checkpoint_sha256': sha(parent_path),
    'protocol_sha256': sha(protocol), 'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'ranker_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'presentations': presentations,
    'scored_pairs': scored, 'seconds': time.perf_counter() - started}), flush=True)
