"""One TRAIN-only no-update equivalence and CE-gradient audit for 128."""

import hashlib
import json
import math
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
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match
from training.in_path_global_matcher_128 import (
    global_plane_match_128, in_path_correspondence_ce_128)

source = Path(__file__).resolve().parent
run = root / 'runs/joint_in_path_correspondence_128_forward_audit'
parent_dir = root / 'runs/joint_pose_map_122'
parent = parent_dir / 'joint_step_02000.pt'
side, seed = 256, 20261010128
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
receipt = json.loads((parent_dir / 'completed.json').read_text())
assert sha(parent) == receipt['checkpoint_sha256']['2000']
context = load_streaming_synthetic_v7_64(device='cuda')
rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
while True:
    virtual = int(torch.randint(len(context['subjects']), (1,), generator=rng))
    with torch.no_grad():
        sample = sample_one_shot_slide_artifacts_v3(context, [virtual], draw_seed,
            side=side)
    draw_seed += 1
    if bool(sample['eligible'][0]):
        break
torch.manual_seed(seed)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(torch.load(parent, map_location='cpu',
    weights_only=True)['model'], strict=True)
prediction = model.predict(sample['inputs'])
prior = (prediction['log_mass'][..., None] + torch.stack((
    F.logsigmoid(-prediction['reflection_logit']),
    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
with torch.no_grad():
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
        prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    row = torch.arange(1, device='cuda')
    for _ in range(2):
        chosen = normals[row[:, None], beam // 2]
        similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity = -similarity
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[row, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    mode, reflection = beam // 2, beam % 2
old = global_plane_match(model, prediction, mode, reflection,
    sample['offsets'], sample['weights'], context['atlas'],
    sample['inputs'].shape[-2:], side=24)
new = global_plane_match_128(model, prediction, mode, reflection,
    sample['offsets'], sample['weights'], context['atlas'],
    sample['inputs'].shape[-2:], side=24)
assert all(torch.equal(old[key], new[key]) for key in old)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
fixed = rigid_points_090(sample['state'], sample['reflection'], corners[None])
candidate = rigid_points_090(old['input_state'], reflection, corners[None, None])
cost = (candidate - fixed[:, None]).norm(dim=-1).mean(-1) / 1000
target = F.softmax(-cost.detach() / .7, -1)
base = F.kl_div(F.log_softmax(old['score'], -1), target,
    reduction='batchmean')
jitter = torch.zeros(1, 9, device='cuda')
jitter[0, 2] = .05 * (2 * torch.rand(1, generator=rng).cuda() - 1)
jitter[0, 3:5] = 750 * (2 * torch.rand(2, generator=rng).cuda() - 1)
jittered = compose_full_frame_state(sample['state'], jitter)
best = cost.argmin(-1)
near_mode = mode.gather(1, best[:, None])
near_reflection = reflection.gather(1, best[:, None])
ce_state = torch.cat((jittered[:, None],
    prediction['state'][torch.arange(1, device='cuda')[:, None], near_mode]), 1)
ce_mode = near_mode.expand(-1, 2)
ce_reflection = torch.cat((sample['reflection'][:, None], near_reflection), 1)
match = global_plane_match_128(model, prediction, ce_mode, ce_reflection,
    sample['offsets'], sample['weights'], context['atlas'],
    sample['inputs'].shape[-2:], side=24, candidate_state=ce_state,
    return_attention=True)
positive = torch.cat((torch.ones_like(best[:, None], dtype=torch.bool),
    cost.gather(1, best[:, None]) <= 1.5), 1)
ce = in_path_correspondence_ce_128(match, ce_state, ce_reflection,
    sample['centre'], sample['valid_mask'], sample['offsets'], positive)
probes = (model.global_plane_matcher_120['source'][0].weight,
    model.global_plane_matcher_120['atlas'][0].weight,
    model.encoder[0][0].weight, model.pose[-1].weight)
base_gradient = torch.autograd.grad(base, probes, retain_graph=True, allow_unused=True)
ce_gradient = torch.autograd.grad(ce['loss'], probes, allow_unused=True)
base_norm = [0. if value is None else float(value.norm()) for value in base_gradient]
ce_norm = [0. if value is None else float(value.norm()) for value in ce_gradient]
assert all(math.isfinite(value) for value in base_norm + ce_norm)
assert all(value > 0 for value in ce_norm[:3])
result = {'draw_seed': draw_seed - 1, 'subject_index': virtual,
    'provenance': sample['provenance'][0],
    'frozen_120_default_exact_equal': True,
    'ce_loss': float(ce['loss']),
    'ce_pixels_jittered': int(ce['pixels_per_candidate'][0, 0]),
    'ce_pixels_predicted_near': int(ce['pixels_per_candidate'][0, 1]),
    'predicted_near_eligible': bool(positive[0, 1]),
    'probe_order': ['matcher_source', 'matcher_atlas', 'shared_encoder', 'direct_pose_head'],
    'base_action_gradient_norm': base_norm, 'ce_gradient_norm': ce_norm,
    'weighted_ce_to_base_action_ratio': [.05 * ce_norm[i] / max(base_norm[i], 1e-12)
        for i in range(4)],
    'parent_checkpoint_sha256': sha(parent),
    'source_sha256': {name: sha(source / name) for name in (
        'audit_joint_in_path_correspondence_128_forward.py',
        'in_path_global_matcher_128.py', 'global_plane_matcher_120.py')},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False,
    'external_pretrained_weights_used': False}
run.mkdir(parents=True)
(run / 'completed.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
