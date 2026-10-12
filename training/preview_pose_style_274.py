"""Four new independent synthetic inputs beside true and image-only predicted atlas planes."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090


RUN = ROOT / 'runs/pose_style_274_train_inner_001'
PARENT = ROOT / 'runs/pose_capture_270_v9_continuation_001'
OUT = ROOT / 'previews/pose_style_274_true_input_predicted_001'
assert Path(__file__).drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
torch.set_num_threads(4)
panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
chosen = [sorted(panel, key=lambda item: item['obliquity_deg'])[index]
          for index in (8, 24, 40, 63)]
assert len({item['physical_section_id'] for item in chosen}) == 4
receipt = json.loads((RUN / 'completed.json').read_text())
parent_receipt = json.loads((PARENT / 'completed.json').read_text())
checkpoint_path = RUN / 'control_v9_step_02000.pt'
with checkpoint_path.open('rb') as stream:
    checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert checkpoint_hash == receipt['checkpoint_sha256']['control_v9_2000']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model'])
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda').float() / 16
fig, axes = plt.subplots(4, 3, figsize=(11, 15))
records = []
with torch.inference_mode():
    for row, item in enumerate(chosen):
        prediction = model.predict(item['inputs'][None].cuda())
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
        branch = int(prior.argmax())
        chosen_state = prediction['state'][0, branch // 2]
        thickness = item['provenance']['thickness_um']
        offsets = torch.linspace(-.5, .5, 9, device='cuda') * thickness
        states = torch.stack((item['state'].cuda(), chosen_state))[None]
        reflections = torch.tensor([[int(item['reflection']), branch % 2]], device='cuda')
        rendered = render_atlas_planes_090(atlas, states, reflections, offsets, weights,
                                           source_shape=(192, 192))[0]
        chart = item['chart'].cuda()
        true = rigid_points_090(item['state'][None].cuda(), item['reflection'][None].cuda(),
                                chart, (192, 192))
        selected = rigid_points_090(chosen_state[None], reflections[:, 1], chart, (192, 192))
        error_mm = float((selected - true).norm(dim=-1).mean() / 1000)
        images = (rendered[0, 0].cpu().numpy(), item['inputs'][0].numpy(),
                  rendered[1, 0].cpu().numpy())
        for column, image in enumerate(images):
            axes[row, column].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[row, column].axis('off')
        axes[row, 0].set_title(f"True atlas | {item['obliquity_deg']:.0f}° from axis")
        axes[row, 1].set_title(f"Generated input | {item['visible_fraction']:.0%} visible")
        axes[row, 2].set_title(f'Predicted atlas | {error_mm:.2f} mm')
        records.append({'physical_section_id': item['physical_section_id'],
            'base_index': item['base_index'], 'obliquity_deg': item['obliquity_deg'],
            'visible_fraction': item['visible_fraction'], 'selected_branch': branch,
            'first_choice_site_error_mm': error_mm,
            'provenance': item['provenance']})
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=False)
fig.savefig(OUT / 'true_input_predicted.png', dpi=140)
plt.close(fig)
(OUT / 'selection.json').write_text(json.dumps({
    'selection_rule': 'sorted 270 fresh-panel obliquity positions 8,24,40,63; no error-based choice',
    'checkpoint_sha256': checkpoint_hash,
    'panel_sha256': parent_receipt['inner_panel_sha256'],
    'render': 'same nine-node normalized finite-thickness atlas PSF for true/predicted',
    'scope': 'synthetic TRAIN-inner qualitative inspection; pose-only, no learned local map',
    'rows': records}, indent=2))
(OUT / 'completed.json').write_text(json.dumps({
    'image_sha256': hashlib.sha256((OUT / 'true_input_predicted.png').read_bytes()).hexdigest(),
    'selection_sha256': hashlib.sha256((OUT / 'selection.json').read_bytes()).hexdigest()}, indent=2))
print(OUT / 'true_input_predicted.png')
