"""Matched source–atlas evidence arms within the standalone joint model."""

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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_evidence_130 import global_plane_evidence_130
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


arm = os.environ['CROSSED130_ARM']
assert arm in ('full', 'support')
source = Path(__file__).resolve().parent
assert root.drive.upper() == source.drive.upper() == 'I:'
run = root / f'runs/crossed_plane_evidence_130_{arm}'
parent = root / 'runs/joint_in_path_correspondence_128_treatment/joint_step_06000.pt'
parent_done = json.loads((parent.parent / 'completed.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['6000']
seed, updates, side = 20261010130, 2000, 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda()
attach_global_plane_matcher(model, enabled=True)
state = torch.load(parent, map_location='cpu', weights_only=True)
assert state['step'] == 6000 and state['calibrated'] is False
model.load_state_dict(state['model'], strict=True)
del state
model.global_plane_matcher_120['evidence'] = nn.Sequential(
    nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)).cuda()
model.eval().requires_grad_(False)
matcher = model.global_plane_matcher_120
for module in (matcher['source'], matcher['atlas'], matcher['head'][:-1], matcher['evidence']):
    module.requires_grad_(True)
old_parameters = [*matcher['source'].parameters(), *matcher['atlas'].parameters(),
                  *matcher['head'][:-1].parameters()]
parameters = [*old_parameters, *matcher['evidence'].parameters()]
optimizer = torch.optim.AdamW([
    {'params': old_parameters, 'lr': 3e-5},
    {'params': matcher['evidence'].parameters(), 'lr': 1e-4}], weight_decay=1e-4)

names = ('train_crossed_plane_evidence_130.py', 'global_plane_evidence_130.py',
    'in_path_global_matcher_128.py', 'global_plane_matcher_120.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py')
config = {'arm': arm, 'seed': seed, 'updates': updates, 'side': side,
    'physical_sections_per_update': 2, 'parent_checkpoint': str(parent),
    'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent.parent / 'completed.json'),
    'synthetic_context_provenance': context['provenance'],
    'source_sha256': {name: sha(source / name) for name in names},
    'objective': 'crossed exact-state source-atlas interaction plus support-matched blind near/wrong posterior margin',
    'stage': 'train evidence head in final joint architecture; direct pose and mapper frozen until evidence gate',
    'loss_weights': {'crossed': 1., 'blind_support_matched': 1.},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
resume = run / 'resume.pt'
if resume.exists():
    assert json.loads((run / 'config.json').read_text()) == config
    saved = torch.load(resume, map_location='cpu', weights_only=True)
    start = saved['step']
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    lines = [line for line in (run / 'training.jsonl').read_text().splitlines()
             if json.loads(line)['update'] <= start]
    (run / 'training.jsonl').write_text('\n'.join(lines) + ('\n' if lines else ''))
    print(json.dumps({'event': 'resumed', 'arm': arm, 'update': start}), flush=True)
else:
    assert not run.exists()
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    start = 0


def save(path, step):
    temp = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'config': config,
        'calibrated': False}, temp)
    os.replace(temp, path)


five = torch.tensor(((0., 0.), (255., 0.), (0., 255.), (255., 255.),
                     (127.5, 127.5)), device='cuda')[None] / side
if start == 0:
    save(run / 'joint_step_00000.pt', 0)
    save(resume, 0)
started = time.perf_counter()
with (run / 'training.jsonl').open('a') as log:
    for step in range(start + 1, updates + 1):
        samples, predictions, draws = [], [], []
        for slot in range(2):
            attempt = 0
            while True:
                draw_seed = seed * 100000000 + step * 1000000 + slot * 100000 + attempt
                virtual = int(np.random.default_rng(draw_seed + 13).integers(
                    len(context['subjects'])))
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v4(context,
                        [virtual], draw_seed, side=side)
                if bool(sample['eligible'][0]):
                    break
                attempt += 1
            assert sample['provenance'][0]['split'] == 'train'
            with torch.no_grad():
                prediction = model.predict(sample['inputs'])
            samples.append(sample)
            predictions.append(prediction)
            draws.append({'draw_seed': draw_seed, 'virtual_index': virtual,
                          'attempts': attempt + 1, 'provenance': sample['provenance'][0]})

        optimizer.zero_grad(set_to_none=True)
        truth_state = torch.cat([sample['state'] for sample in samples])
        truth_reflection = torch.cat([sample['reflection'] for sample in samples])
        crossed_prediction = {key: torch.cat([prediction[key] for prediction in predictions])
            for key in ('feature', 'log_mass', 'reflection_logit')}
        crossed = global_plane_evidence_130(model, crossed_prediction,
            torch.zeros((2, 2), device='cuda', dtype=torch.long),
            truth_reflection[None].expand(2, -1),
            torch.cat([sample['offsets'] for sample in samples]),
            torch.cat([sample['weights'] for sample in samples]),
            context['atlas'], (side, side),
            candidate_state=truth_state[None].expand(2, -1, -1),
            support_only=(arm == 'support'))
        evidence = crossed['evidence_logit']
        interaction = evidence[0, 0] + evidence[1, 1] - evidence[0, 1] - evidence[1, 0]
        crossed_loss = F.softplus(1 - interaction)

        hard_losses, selected_costs, near_available = [], [], 0
        for sample, prediction in zip(samples, predictions):
            with torch.no_grad():
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                    prior[:, 32:].topk(6, -1).indices + 32), -1)
                normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
                for _ in range(2):
                    chosen = normals[0, beam[0] // 2]
                    similarity = (normals[0, 16:, None] * chosen[None]).sum(-1).abs().amax(-1)
                    diversity = -similarity
                    diversity.scatter_(0, beam[0, 8:] // 2 - 16, -2.)
                    anchor = diversity.argmax()
                    reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax()
                    beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[None, None]), -1)
                mode, reflection = beam // 2, beam % 2
                states = prediction['state'][0, mode[0]][None]
                target = rigid_points_090(sample['state'], sample['reflection'], five)
                points = rigid_points_090(states, reflection, five[:, None])
                normal = full_frame_state_to_components(states)[1][..., :, 2]
                true_normal = full_frame_state_to_components(sample['state'])[1][..., :, 2]
                cost = ((points - target[:, None]).norm(dim=-1).mean(-1) / 1000
                    + 4 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1)))
                best = int(cost[0].argmin())
                near_available += int(cost[0, best] <= 1.5)
            matched = global_plane_evidence_130(model, prediction, mode, reflection,
                sample['offsets'], sample['weights'], context['atlas'], (side, side),
                support_only=(arm == 'support'), return_support=True)
            score = matched['original_score'][0]
            selected_costs.append(float(cost[0, score.argmax()].detach()))
            if cost[0, best] <= 1.5:
                with torch.no_grad():
                    support = matched['atlas_support'][0].detach()
                    mask = support > .5
                    near_mask = mask[best]
                    dice = 2 * (mask & near_mask).sum((-2, -1)) / (
                        mask.sum((-2, -1)) + near_mask.sum()).clamp_min(1)
                    intact = F.adaptive_avg_pool2d(sample['valid_mask'][:, None].float(),
                        (24, 24))[0, 0] >= .95
                    common = (intact & (support > .8) & (support[best] > .8))
                    eligible = ((cost[0] >= cost[0, best] + 1.) & (dice >= .8)
                        & (common.sum((-2, -1)) >= 32))
                    wrong = int(prior[0, beam[0]].masked_fill(~eligible, -1e9).argmax())
                if bool(eligible[wrong]):
                    hard_losses.append(F.softplus(1 - score[best] + score[wrong]))

        hard_loss = (torch.stack(hard_losses).mean() if hard_losses
                     else crossed_loss * 0)
        loss = crossed_loss + hard_loss
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'draws': draws,
            'crossed_interaction': float(interaction.detach()),
            'crossed_loss': float(crossed_loss.detach()),
            'support_matched_blind_pairs': len(hard_losses),
            'blind_hard_loss': float(hard_loss.detach()),
            'blind_near_available': near_available,
            'blind_selected_rigid_cost_mm': float(np.mean(selected_costs)),
            'gradient_norm': float(gradient),
            'seconds_since_process_start': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 250 == 0:
            print(json.dumps({'event': 'train_milestone', 'arm': arm,
                **{key: row[key] for key in ('update', 'crossed_interaction',
                    'support_matched_blind_pairs', 'blind_near_available',
                    'blind_selected_rigid_cost_mm', 'gradient_norm')}}), flush=True)
        if step % 100 == 0:
            log.flush()
            save(resume, step)
        if step in (500, 1000, 2000):
            log.flush()
            save(run / f'joint_step_{step:05d}.pt', step)

(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'accepted_synthetic': 2 * updates,
    'parent_checkpoint_sha256': sha(parent),
    'config_sha256': sha(run / 'config.json'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
        for step in (0, 500, 1000, 2000)},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'train_complete', 'arm': arm, 'updates': updates,
    'seconds_since_process_start': time.perf_counter() - started}), flush=True)
