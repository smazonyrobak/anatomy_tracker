"""One frozen TRAIN-only framing check with six actual-generator examples."""
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['MPLCONFIGDIR'] = str(root / 'cache/matplotlib')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090

source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/FOV_BALANCE_V10_PREFLIGHT_PROTOCOL_20261014.md'
generator = source.with_name('arbitrary_plane_fov_balance_v10.py')
parent = root / 'runs/joint_spatial_305_acquired_rehearsal_001/intervention/pose_local_00400.pt'
baseline = root / 'runs/blind_ranking_only_306_001/panels.pt'
output = root / 'runs/fov_balance_v10_preflight_001'
assert source.drive.upper() == output.drive.upper() == 'I:' and not output.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_fov_balance_v10('cuda')
model, _ = load_one_shot_checkpoint(parent, device='cuda')
seed = 20261014310
rng = np.random.default_rng(seed)
fig, axes = plt.subplots(6, 3, figsize=(9, 16), layout='constrained')
config = {'seed': seed, 'train_inner_bases': list(range(56, 64)),
    'cases_per_base': 8, 'fov_scale': .72,
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'generator_sha256': hashlib.sha256(generator.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'checkpoint_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
    'baseline_panel_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
    'generator_provenance': context['provenance'],
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
output.mkdir(parents=True, exist_ok=False)
(output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
rows = []
with (output / 'draws.jsonl').open('w', encoding='utf8') as draws, torch.inference_mode():
    for base in range(56, 64):
        accepted = attempted = 0
        while accepted < 8:
            attempted += 1
            variant = int(rng.integers(VARIANTS))
            draw_seed = int(rng.integers(2**63 - 1, dtype=np.int64))
            sample = sample_fov_balance_v10(context,
                [base * VARIANTS + variant], draw_seed, 192)
            used = bool(sample['eligible'][0])
            draws.write(json.dumps({'base': base, 'attempt': attempted,
                'accepted_slot': accepted if used else None, 'variant': variant,
                'seed': draw_seed, 'accepted': used, 'provenance': sample['provenance'][0]},
                allow_nan=False) + '\n')
            if not used:
                continue
            accepted += 1
            prediction = model.predict(sample['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            branch = int(prior.argmax())
            state = prediction['state'][:, branch // 2, None]
            reflection = torch.tensor([[branch % 2]], device='cuda')
            guessed = render_atlas_planes_090(context['atlas'], state, reflection,
                sample['offsets'], sample['weights'], candidate_chunk=1,
                source_shape=(192, 192))[0, 0]
            valid = sample['valid_mask'][0]
            sites = torch.multinomial(valid.flatten().float(), 128, replacement=True,
                generator=torch.Generator(device='cuda').manual_seed(draw_seed))
            chart = torch.stack((sites.remainder(192),
                sites.div(192, rounding_mode='floor')), -1).float() / 192
            truth = sample['centre'][0].reshape(-1, 3)[sites]
            predicted = rigid_points_090(state[0], reflection[0], chart, (192, 192))[0]
            error_mm = float((predicted - truth).norm(dim=-1).mean() / 1000)
            normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
            obliquity = math.degrees(math.acos(float(normal.abs().max().clamp(0, 1))))
            record = sample['provenance'][0]
            row = {'physical_section_id': record['physical_section_id'],
                'base': base, 'variant': variant, 'seed': draw_seed,
                'framing': record['framing']['kind'],
                'sampling_branch': record['sampling_branch'],
                'exterior_style': record['one_shot_slide_artifacts_v9']['exterior_style'],
                'visible_fraction': float(valid.float().mean()),
                'obliquity_deg': obliquity, 'first_rigid_error_mm': error_mm}
            rows.append(row)
            if len(rows) <= 6:
                tissue = sample['clean'][0] / sample['support'][0].clamp_min(1e-4)
                tissue = torch.where(sample['support'][0] > .05, tissue, 0)
                observed = sample['inputs'][0, 0]
                p99 = max(float(torch.quantile(observed.flatten(), .99)), .001)
                atlas = torch.where(guessed[1] > .05, guessed[0], 0)
                for column, image in enumerate((tissue, observed / p99, atlas)):
                    axes[len(rows) - 1, column].imshow(image.cpu(), cmap='gray', vmin=0, vmax=1)
                    axes[len(rows) - 1, column].axis('off')
                axes[len(rows) - 1, 0].text(.02, .98, f'{obliquity:.0f}° oblique',
                    transform=axes[len(rows) - 1, 0].transAxes, va='top', color='white',
                    fontsize=9, bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none'})
                axes[len(rows) - 1, 1].text(.02, .98,
                    f'{row["exterior_style"]}; tissue {row["visible_fraction"]:.0%}',
                    transform=axes[len(rows) - 1, 1].transAxes, va='top', color='white',
                    fontsize=9, bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none'})
                axes[len(rows) - 1, 2].text(.02, .98, f'{error_mm:.2f} mm site error',
                    transform=axes[len(rows) - 1, 2].transAxes, va='top', color='white',
                    fontsize=9, bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none'})
            if len(rows) % 16 == 0:
                draws.flush()
                print(json.dumps({'event': 'case_milestone', 'accepted': len(rows)}), flush=True)
assert len(rows) == 64 and len({row['physical_section_id'] for row in rows}) == 64
for axis, title in zip(axes[0], ('True sampled atlas texture',
        'Actual generated input\n(display stretched)', 'Atlas at blind guessed plane')):
    axis.set_title(title)
fig.savefig(output / 'first_six.png', dpi=150)
plt.close(fig)
(output / 'cases.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n'
    for row in rows), encoding='utf8')

baseline_panel = torch.load(baseline, map_location='cpu', weights_only=True)
old_fraction = [row['visible_fraction'] for group in baseline_panel.values() for row in group]
archive = root / 'data/lupori_joint_185/rawData.zip'
real_fraction = []
with ZipFile(archive) as files:
    for animal in ('CC2B', 'CC4B', 'CC4A'):
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            files.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        registration = json.loads(files.read(f'{animal}/{animal}-visualign.json'))
        for item in registration['slices']:
            section = Path(item['filename']).name.removesuffix('-thumb.png')
            if section in valid:
                mask = np.asarray(Image.open(io.BytesIO(files.read(
                    f'{animal}/masks/{section}-mask.png'))).convert('L').resize(
                    (192, 192), Image.Resampling.NEAREST)) > 127
                real_fraction.append(float(mask.mean()))
assert len(real_fraction) == 188
fraction = [row['visible_fraction'] for row in rows]
quantiles = {name: np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist()
    for name, values in (('v9_synthetic', old_fraction),
                         ('v10_synthetic', fraction), ('fluorescence_train', real_fraction))}
gate = (.35 <= float(np.median(fraction)) <= .65
    and sum(value < .15 for value in fraction) >= 4
    and sum(row['obliquity_deg'] >= 45 for row in rows) >= 4)
summary = {'draws': sum(1 for _ in (output / 'draws.jsonl').open(encoding='utf8')),
    'accepted': len(rows), 'fraction_quantiles': quantiles,
    'v10_sparse_lt15pct': sum(value < .15 for value in fraction),
    'v10_obliquity_ge45': sum(row['obliquity_deg'] >= 45 for row in rows),
    'v10_first_error_mean_mm_untrained_domain': float(np.mean([
        row['first_rigid_error_mm'] for row in rows])),
    'framing_feasibility_gate_pass': gate,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
(output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(output / 'completed.json').write_text(json.dumps({'summary': summary,
    'output_sha256': {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in output.iterdir() if file.name != 'completed.json'}}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'gate_pass': gate,
    'v10_median_fraction': quantiles['v10_synthetic'][3],
    'v9_median_fraction': quantiles['v9_synthetic'][3],
    'real_median_fraction': quantiles['fluorescence_train'][3]}), flush=True)
