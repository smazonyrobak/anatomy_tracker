"""One TRAIN case: prove frozen atlas fitting can backpropagate into pose geometry."""

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

from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_156 import JointAnatomyModel156


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
seed = 2026101115601
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
train = root / 'runs/joint_fitter_155_train151_001'
done = json.loads((train / 'completed.json').read_text())
step = done['selected_step']
checkpoint = train / f'joint_fitter_step_{step:05d}.pt'
assert sha(checkpoint) == done['checkpoint_sha256'][str(step)]
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = JointAnatomyModel156().cuda().eval()
model.load_state_dict(saved['model'], strict=True)

for attempt in range(1, 101):
    base = int(rng.integers(60))
    draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    with torch.no_grad():
        original = sample_panel_geometry_train_151(context, [base], draw_seed, 256)
        sample = sample_one_shot_slide_artifacts_v4(
            context, [base], draw_seed, 256, source_sample=original)
        if not bool(sample['eligible'][0]):
            continue
        pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
            device='cuda').long()]
        chart = torch.stack((pixels.remainder(256),
            pixels.div(256, rounding_mode='floor')), -1).float() / 256
        prediction = model.pose.predict(sample['inputs'])
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = prior.topk(16, -1).indices[0]
        proposed = rigid_points_090(prediction['state'][0, branch // 2],
                                    branch % 2, chart)
        truth = rigid_points_090(sample['state'], sample['reflection'], chart)
        error = (proposed - truth).norm(dim=-1).mean(-1) / 1000
        best = int(error.argmin())
        if error[best] <= 1.5:
            choice = branch[best:best + 1][None]
            break
else:
    raise RuntimeError('No near on-policy TRAIN case among 100 independent draws')

mode = choice.div(2, rounding_mode='floor')
reflection = choice.remainder(2)
output = model(sample['inputs'], context['atlas'], sample['offsets'],
    sample['weights'], mode, reflection)
corrected = rigid_points_090(output['corrected_state'].float(), reflection, chart[None])
loss = F.smooth_l1_loss((corrected - truth[:, None]).norm(dim=-1).mean(-1) / 1000,
                        torch.zeros(1, 1, device='cuda'), beta=.25)
targets = (output['candidate_state'], output['matcher']['expected_ccf'],
           model.pose.encoder[0][0].weight, model.matcher.atlas[0].weight)
grads = torch.autograd.grad(loss, targets, allow_unused=True)
full_norms = [None if grad is None else float(grad.norm()) for grad in grads]

control = model(sample['inputs'], context['atlas'], sample['offsets'],
    sample['weights'], mode, reflection, state_override=output['candidate_state'].detach())
control_corrected = rigid_points_090(control['corrected_state'].float(), reflection, chart[None])
control_loss = F.smooth_l1_loss((control_corrected - truth[:, None]).norm(dim=-1).mean(-1) / 1000,
                                torch.zeros(1, 1, device='cuda'), beta=.25)
control_grad = torch.autograd.grad(control_loss, model.pose.encoder[0][0].weight,
                                  allow_unused=True)[0]
result = {'base_index': base, 'physical_section_id': sample['provenance'][0]['physical_section_id'],
    'draw_seed': draw_seed, 'attempt': attempt, 'prefit_mm': float(error[best]),
    'postfit_loss': float(loss.detach()), 'control_loss': float(control_loss.detach()),
    'gradient_norms': dict(zip(('candidate_state', 'expected_ccf', 'pose_encoder',
                                'atlas_descriptor'), full_norms)),
    'control_pose_gradient_norm': None if control_grad is None else float(control_grad.norm()),
    'checkpoint_sha256': sha(checkpoint), 'source_sha256': sha(__file__)}
assert abs(result['postfit_loss'] - result['control_loss']) < 1e-6
assert all(value is not None and np.isfinite(value) and value > 0 for value in full_norms)
assert result['control_pose_gradient_norm'] is None or result['control_pose_gradient_norm'] == 0
print(json.dumps(result, allow_nan=False), flush=True)
