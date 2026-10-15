"""Four frozen TRAIN cuts: correct atlas plane, actual synthetic image, C4-selected plane."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from training.arbitrary_plane_c4_joint_model import C4JointSliceModel
from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import render_atlas_planes_090

torch.set_num_threads(4)
run = ROOT / 'runs/c4_joint_321_002'
panel = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)
readout = json.loads((run / 'readouts.jsonl').read_text().splitlines()[-1])
rows = {(row['cohort'], row['physical_section_id']): row for row in readout['rows']
        if 'physical_section_id' in row}
checkpoint = torch.load(run / 'joint_12000.pt', map_location='cpu', weights_only=True)
model = C4JointSliceModel(side=192).cuda().eval()
model.load_state_dict(checkpoint['model'])
context = load_fov_balance_v10('cuda')
chosen = [('inner', 0), ('inner', 64), ('steep', 0), ('sparse', 0)]
figure, axes = plt.subplots(4, 3, figsize=(11, 14), constrained_layout=True)
for row, (cohort, index) in enumerate(chosen):
    case = panel[cohort][index]
    meta = case['meta']
    with torch.inference_mode():
        sample = sample_fov_balance_v10(context,
            [meta['base'] * VARIANTS + meta['variant']], meta['seed'], 192)
        assert sample['provenance'][0]['physical_section_id'] == meta['physical_section_id']
        assert torch.allclose(sample['inputs'][0].cpu(), case['input'], atol=1e-5)
        prediction = model.predict(sample['inputs'])
        scores = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(scores[0].argmax())
        states = torch.stack((sample['state'][0], prediction['state'][0, branch // 2]))[None]
        flips = torch.tensor([[int(sample['reflection'][0]), branch % 2]], device='cuda')
        rendered = render_atlas_planes_090(context['atlas'], states, flips,
            sample['offsets'], sample['weights'], source_shape=(192, 192))[0, :, 0]
        rendered = torch.stack((rendered[0], torch.rot90(rendered[1],
                                 branch // (2 * model.normal_modes), (-2, -1))))
        atlas = F.interpolate(rendered[:, None], (192, 192), mode='bilinear',
                              align_corners=False)[:, 0].cpu().numpy()
        images = (atlas[0], case['input'][0].numpy(), atlas[1])
    for col, image in enumerate(images):
        axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[row, col].axis('off')
    error = rows[(cohort, meta['physical_section_id'])]['first_rigid_gap_mm']
    axes[row, 0].text(.02, .98, f'{cohort} · {meta["obliquity_deg"]:.0f}° obliquity\n'
                           f'blind error {error:.2f} mm', transform=axes[row, 0].transAxes,
                           ha='left', va='top', color='white', fontsize=9,
                           bbox={'facecolor': 'black', 'alpha': .7, 'edgecolor': 'none'})
for axis, title in zip(axes[0], ('Correct atlas plane', 'Actual damaged synthetic input',
                                 'Blind C4-selected atlas plane')):
    axis.set_title(title)
figure.suptitle('Frozen TRAIN synthetic cuts; fixed indices, not acquired-animal validation')
destination = Path(__file__).resolve().parent.parent / \
    'docs/publication/figures/c4_321_actual_generator_examples.png'
destination.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(destination, dpi=150)
print(destination)
