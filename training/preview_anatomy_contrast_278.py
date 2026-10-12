"""Display six distinct v11 generator sections and their atlas planes."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['MPLCONFIGDIR'] = str(root / 'cache/matplotlib')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_slide_artifacts_v11 import sample_one_shot_slide_artifacts_v11
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/ANATOMY_CONTRAST_278_GENERATOR_PROTOCOL_20261012.md'
out = root / 'previews/anatomy_contrast_278_actual_generator_001'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
atlas_array, annotation_array = _decode_and_preprocess_allen_v6()
annotation = torch.from_numpy(annotation_array.astype(np.int64)).cuda()
del atlas_array, annotation_array
rng = np.random.default_rng(20261012278)
accepted, draws = [], []
with torch.inference_mode():
    while len(accepted) < 16:
        attempt = len(draws) + 1
        base = int(rng.integers(56))
        subject = base * VARIANTS + int(rng.integers(VARIANTS))
        seed = 20261012278 + 1_000_000 + attempt
        sample = sample_one_shot_slide_artifacts_v11(context, annotation, [subject], seed, 192)
        used = bool(sample['eligible'][0])
        draws.append({'attempt': attempt, 'subject_index': subject, 'seed': seed,
                      'used': used, 'physical_section_id':
                      sample['provenance'][0]['physical_section_id']})
        if used:
            state = sample['state'][0]
            normal = full_frame_state_to_components(state)[1][:, 2]
            obliquity = float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1))))
            atlas = render_atlas_planes_090(context['atlas'], state[None, None],
                sample['reflection'][0][None, None], sample['offsets'][0],
                sample['weights'][0], source_shape=(192, 192))[0, 0, 0]
            accepted.append({'obliquity_deg': obliquity,
                'atlas': atlas.cpu().numpy(),
                'undamaged': sample['undamaged_image'][0].cpu().numpy(),
                'input': sample['inputs'][0, 0].cpu().numpy(),
                'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                'provenance': sample['provenance'][0]})
        del sample
selected = [sorted(accepted, key=lambda item: item['obliquity_deg'])[i]
            for i in (0, 3, 6, 9, 12, 15)]
fig, axes = plt.subplots(6, 3, figsize=(11, 20))
for row, item in enumerate(selected):
    for column, name in enumerate(('atlas', 'undamaged', 'input')):
        axes[row, column].imshow(item[name], cmap='gray', vmin=0, vmax=1)
        axes[row, column].axis('off')
    axes[row, 0].set_title(f"Fitted atlas | {item['obliquity_deg']:.0f}° from axis")
    axes[row, 1].set_title('Styled source before damage')
    axes[row, 2].set_title(f"Final model input | {item['visible_fraction']:.0%} tissue")
fig.tight_layout()
out.mkdir(parents=True, exist_ok=False)
fig.savefig(out / 'atlas_source_input.png', dpi=140)
plt.close(fig)
(out / 'selection.json').write_text(json.dumps({
    'rule': '16 independently sampled eligible planes, sorted by obliquity; positions 0,3,6,9,12,15',
    'draws': draws, 'rows': [{key: value for key, value in item.items()
                            if key not in ('atlas', 'undamaged', 'input')}
                           for item in selected],
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'generator_sha256': hashlib.sha256(source.with_name(
        'arbitrary_plane_one_shot_slide_artifacts_v11.py').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'display': 'all images raw model intensities, grayscale vmin=0 vmax=1',
    'atlas_render': 'nine-node normalized finite-thickness fitted affine CCF plane; local subject warp remains in synthetic input'
}, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'image_sha256': hashlib.sha256((out / 'atlas_source_input.png').read_bytes()).hexdigest(),
    'selection_sha256': hashlib.sha256((out / 'selection.json').read_bytes()).hexdigest()}, indent=2))
print(json.dumps({'event': 'complete', 'path': str(out / 'atlas_source_input.png'),
                  'draws': len(draws), 'rows': len(selected)}), flush=True)
