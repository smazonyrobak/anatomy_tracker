"""First six independent v7 draws: true atlas, input, and blind 244 fit."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
os.environ['MPLCONFIGDIR'] = str(root / 'cache/matplotlib')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, normalized_raster_to_ccf,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rng = np.random.default_rng(20261012246)
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
samples = []
while len(samples) < 6:
    subject = int(rng.integers(len(context['subjects'])))
    seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    with torch.inference_mode():
        sample = sample_one_shot_slide_artifacts_v7(context, [subject], seed, side=256)
    if bool(sample['eligible'][0]):
        samples.append(sample)

checkpoint = root / 'runs/shared_anchor_joint_feedback_244_train_001/joint_step_06000.pt'
with checkpoint.open('rb') as stream:
    checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, shared_anchor_pose=True,
                              atlas_conditioning=True, vector_refinement=True,
                              fitted_ranking=True).cuda().eval().requires_grad_(False)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)
del saved

axis = torch.arange(256, device='cuda') / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
figure, axes = plt.subplots(6, 3, figsize=(11, 18), constrained_layout=True)
rows = []
with torch.inference_mode():
    for i, sample in enumerate(samples):
        image = sample['inputs']
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        choice = prior.topk(8).indices
        state = prediction['state'][:, choice // 2]
        reflection = (choice % 2)[None]
        selected = {**prediction, 'state': state,
                    'log_mass': prediction['log_mass'][:, choice // 2],
                    'reflection_logit': prediction['reflection_logit'][:, choice // 2]}
        index = torch.arange(8, device='cuda')[None]
        first = model.map(selected, sample['offsets'], index, reflection, (64, 64),
                          atlas, sample['weights'], return_refinement_feature=True,
                          feature_side=64, source_shape=(256, 256))
        refined_state, delta, _ = model.refine(first['refinement_feature'], state)
        refined = {**selected, 'state': refined_state}
        mapped = model.map(refined, sample['offsets'], index, reflection, (96, 96),
                           atlas, sample['weights'], feature_side=96,
                           source_shape=(256, 256))
        fitted = model.score_fitted_candidates(
            image, refined, mapped, atlas, sample['weights'])[0] + delta[0]
        branch = int(fitted.argmax())
        guess = refined_state[:, branch]
        guess_reflection = reflection[:, branch]

        renders = []
        for plane_state, reflected in ((sample['state'], sample['reflection']),
                                       (guess, guess_reflection)):
            centre, frame, basis = full_frame_state_to_components(plane_state)
            sx = (255 / 256 - xx) if bool(reflected[0]) else xx
            chart = torch.stack((sx, yy), -1)[None]
            plane = normalized_raster_to_ccf(
                centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
            coordinates = plane[:, None] + sample['offsets'][:, :, None, None, None] * frame[
                :, None, None, None, :, 2]
            slab = render_finite_thickness_coordinate_grid(
                atlas, coordinates, (0., 0., 0.), (25., 25., 25.), sample['weights'])[0]
            render = (slab[0] / slab[1].clamp_min(1e-4)).cpu().numpy()
            render[slab[1].cpu().numpy() < .05] = 0
            renders.append(render)

        valid = sample['valid_mask'][0]
        truth = sample['centre'][0, valid]
        centre, frame, basis = full_frame_state_to_components(guess)
        chart = torch.stack((xx[valid], yy[valid]), -1)
        chart[:, 0] = (255 / 256 - chart[:, 0]) if bool(guess_reflection[0]) else chart[:, 0]
        rigid = centre[:, None] + torch.einsum(
            'bij,pj->bpi', frame[..., :2] @ basis, chart - .5)
        error_mm = float((rigid[0] - truth).norm(dim=-1).mean() / 1000)
        record = sample['provenance'][0]
        normal = np.asarray(record['virtual_unit_normal'])
        angle = float(np.degrees(np.arccos(np.clip(np.abs(normal).max(), 0, 1))))
        flags = [name for name, enabled in
                 record['one_shot_slide_artifacts_v3']['events'].items() if enabled]
        for j, panel_image in enumerate((renders[0], image[0, 0].cpu().numpy(), renders[1])):
            axes[i, j].imshow(panel_image, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(
            f'{angle:.0f}° from cardinal\n{record["mode"]}\n{", ".join(flags) or "no added artifact"}',
            fontsize=9)
        axes[i, 2].set_xlabel(f'blind rigid error {error_mm:.2f} mm', fontsize=9)
        rows.append({'physical_section_id': record['physical_section_id'],
                     'virtual_subject': record['virtual_subject_id'],
                     'angle_from_cardinal_deg': angle,
                     'appearance_mode': record['mode'], 'artifact_flags': flags,
                     'fitted_choice_branch': int(choice[branch]),
                     'blind_rigid_error_mm': error_mm})

for axis_obj, title in zip(axes[0], ('Atlas at true plane', 'Generated input',
                                     'Atlas at blind fitted plane')):
    axis_obj.set_title(title)
out = root / 'previews/shared_anchor_v7_true_input_guessed_246.png'
out.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(out, dpi=135)
plt.close(figure)
(out.with_suffix('.json')).write_text(json.dumps({
    'checkpoint_sha256': checkpoint_sha,
    'selection': 'first six eligible independent random v7 draws; no error-based selection',
    'rows': rows}, indent=2))
print(json.dumps({'image': str(out), 'rows': rows}, indent=2))
