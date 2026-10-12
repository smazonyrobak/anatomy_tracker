"""Four fixed held-out v8 sections: correct atlas, damaged input, blind model atlas."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v8 import sample_one_shot_slide_artifacts_v8
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090

torch.set_num_threads(4)
run = ROOT / 'runs/pose_capture_268_v8_train_inner_002'
receipt = json.loads((run / 'completed.json').read_text())
assert receipt['candidate'] == 268
draws = [json.loads(line) for line in (run / 'inner_draws.jsonl').open()]
fixed = [next(row for row in draws if row['used'] and row['base_index'] == base and row['slot'] == 0)
         for base in range(56, 60)]
panel = torch.load(run / 'inner_panel.pt', map_location='cpu', weights_only=False)
checkpoint = torch.load(run / f"pose_step_{receipt['selected_step']:05d}.pt",
                        map_location='cpu', weights_only=False)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
model.load_state_dict(checkpoint['model'])
context = load_streaming_synthetic_v7_64(device='cuda')
figure, axes = plt.subplots(4, 3, figsize=(12, 15), constrained_layout=True)
for row, draw in enumerate(fixed):
    with torch.inference_mode():
        sample = sample_one_shot_slide_artifacts_v8(context, [draw['subject_index']],
                                                     draw['draw_seed'], 192)
        saved = next(item for item in panel if item['base_index'] == draw['base_index']
                     and item['slot'] == 0)
        assert sample['provenance'][0]['physical_section_id'] == saved['physical_section_id']
        assert torch.allclose(sample['inputs'][0].cpu(), saved['inputs'], atol=1e-5)
        prediction = model.predict(sample['inputs'])
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
        selected = int(prior.argmax())
        states = torch.stack((sample['state'][0], prediction['state'][0, selected // 2]))[None]
        flags = torch.tensor([[int(sample['reflection'][0]), selected % 2]], device='cuda')
        rendered = render_atlas_planes_090(context['atlas'], states, flags,
            sample['offsets'], sample['weights'], source_shape=(192, 192))[0, :, 0]
        atlas_images = F.interpolate(rendered[:, None], (192, 192),
                                      mode='bilinear', align_corners=False)[:, 0]
        valid = sample['valid_mask'][0].flatten().nonzero().flatten()
        chosen = valid[torch.linspace(0, len(valid) - 1, min(1024, len(valid)),
                                       device='cuda').long()]
        chart = torch.stack((chosen.remainder(192), chosen.div(192, rounding_mode='floor')), -1).float() / 192
        truth = rigid_points_090(sample['state'], sample['reflection'], chart, (192, 192))
        guess = rigid_points_090(states[:, 1], flags[:, 1], chart, (192, 192))
        error_mm = float((guess - truth).norm(dim=-1).mean() / 1000)
        true_normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
        angle = float(torch.rad2deg(torch.acos(true_normal.abs().max().clamp(0, 1))))
        images = (atlas_images[0].cpu().numpy(), sample['inputs'][0, 0].cpu().numpy(),
                  atlas_images[1].cpu().numpy())
    for column, image in enumerate(images):
        axes[row, column].imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[row, column].axis('off')
    axes[row, 0].text(.02, .98,
        f"base {draw['base_index']} · {angle:.0f}° from cardinal\nselected site error {error_mm:.2f} mm",
        transform=axes[row, 0].transAxes, ha='left', va='top', color='white', fontsize=9,
        bbox={'facecolor': 'black', 'alpha': .7, 'edgecolor': 'none'})
for axis, title in zip(axes[0], ('Correct atlas plane', 'Damaged synthetic input',
                                 'Blind model-selected atlas plane')):
    axis.set_title(title)
figure.suptitle('Four fixed held-out synthetic sections · v8 · not acquired animals')
destination = ROOT / 'reports/pose_capture_268_v8_examples.png'
destination.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(destination, dpi=150)
print(destination)
