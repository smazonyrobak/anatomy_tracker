"""Show five preselected generator cases and the blind plane predictions."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.run_smooth_image_fit_295 import rigid_render

RUN = ROOT / 'reports/blind_candidate_smooth_fit_296_001'
OUT = Path(__file__).resolve().parents[1] / 'docs/publication/figures/blind_candidate_fit_296_examples.png'
panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
rows = [json.loads(line) for line in (RUN / 'sections.jsonl').open(encoding='utf8')]
chosen = []
for event in (None, 'fragment', 'fold', 'bubble', 'tile_seam'):
    chosen.append(next(i for i, item in enumerate(panel)
        if i not in chosen and ((not any(item['event_flags'].values()) and
            item['provenance']['one_shot_slide_artifacts_v9']['exterior_style'] == 'gray') if event is None
                            else item['event_flags'][event])))

context = load_streaming_synthetic_v7_64('cuda')
fig, axes = plt.subplots(5, 4, figsize=(12, 15), constrained_layout=True)
for row_index, index in enumerate(chosen):
    item, result = panel[index], rows[index]
    fitted_index = result['selectors']['fitted_image']['candidate_index']
    states = torch.cat((item['true_state'], item['candidate_states'][:1],
                        item['candidate_states'][fitted_index:fitted_index + 1])).cuda()
    reflections = torch.cat((item['true_reflection'], item['candidate_reflections'][:1],
                             item['candidate_reflections'][fitted_index:fitted_index + 1])).cuda()
    with torch.inference_mode():
        rendered, _, _, _ = rigid_render(context['atlas'], states, reflections,
            item['offsets'].expand(3, -1).cuda(), item['weights'].expand(3, -1).cuda())
    rendered = rendered.cpu().numpy()
    planes = [np.where(x[1] > .25, x[0] / np.maximum(x[1], 1e-4), 0) for x in rendered]
    images = [item['observed_image'][0, 0].numpy(), *planes]
    titles = ['Generated slide (model input)', 'True atlas plane',
              f"Model first plane ({result['selectors']['prior_first']['rigid_site_error_mm']:.2f} mm)",
              f"Fit-score plane ({result['selectors']['fitted_image']['rigid_site_error_mm']:.2f} mm)"]
    event = 'gray background; no event' if row_index == 0 else (None, 'fragment', 'fold', 'bubble', 'tile seam')[row_index]
    for column, (image, title) in enumerate(zip(images, titles)):
        ax = axes[row_index, column]
        ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if column == 0:
            ax.set_ylabel(f"{event}; {item['obliquity_deg']:.0f}° oblique\ncase {index}", fontsize=9)
fig.suptitle('296 TRAIN-inner examples selected by artifact event, not accuracy; no fitted warp shown', fontsize=12)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160)
print(OUT)
print('Case indices:', chosen)
