"""Blind red-donor virtual-cut readout for the frozen 312 paired feedback pilot."""
import hashlib
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, CASES = 20261015313, 192, 64
RUN = ROOT / 'reports/paired_map_to_pose_312_heldout_red'
PILOT = ROOT / 'runs/paired_map_to_pose_312_001'
SOURCES = (
    (257636467, 607916488, 182849231, '47036a75437b94dd0e5b13ea331c74e17a02de8a1fea624a4b3feb31e9cf3684'),
    (526501457, 527105701, 517337553, 'c02a384d34066a3994452758c8ca499fb0461765ac106c52120116d3ac4ca196'),
)
CHECKPOINTS = {
    'parent_309': ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt',
    'detached_control': PILOT / 'detached_control/pose_local_00256.pt',
    'joint_feedback': PILOT / 'joint_feedback/pose_local_00256.pt',
}
assert Path(__file__).resolve().drive.upper() == RUN.drive.upper() == 'I:'
assert not RUN.exists() and (PILOT / 'summary.json').exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

context = load_fov_balance_v10('cuda')
atlas = context['atlas'].clone()
brain = (atlas[1].cpu().numpy() > 0)
panel, bindings = [], []
for experiment, file_id, donor, expected in SOURCES:
    path = ROOT / f'data/allen_stpt_red_310_001/{experiment}_{file_id}.zip'
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    with ZipFile(path) as source:
        header = source.read('resampled_red.mhd').decode()
        assert 'DimSize = 528 320 456' in header and 'ElementSpacing = 25 25 25' in header
        red = np.frombuffer(source.read('resampled_red.raw'), dtype='<u2').reshape(
            (456, 320, 528)).transpose(2, 1, 0)
    low, high = np.quantile(red[brain], (.01, .99))
    image = np.clip((red.astype(np.float32) - low) / (high - low), 0, 1)
    image[~brain] = 0
    context['atlas'][0].copy_(torch.from_numpy(image).cuda())
    bindings.append({'experiment_id': experiment, 'well_known_file_id': file_id,
                     'donor_id': donor, 'archive_sha256': expected,
                     'brain_q01_q99': [float(low), float(high)], 'role': 'DEV clean; no gradients'})
    rng = np.random.default_rng(SEED + donor)
    sites_rng = torch.Generator(device='cuda').manual_seed(SEED + donor + 1)
    accepted = 0
    while accepted < CASES:
        base = int(rng.integers(56, 64))
        variant = int(rng.integers(VARIANTS))
        seed = int(rng.integers(2**63 - 1))
        with torch.inference_mode():
            sample = sample_fov_balance_v10(context, [base * VARIANTS + variant], seed, SIDE)
        if not bool(sample['eligible'][0]):
            continue
        pixels = torch.multinomial(sample['valid_mask'].flatten(1).float(), 128,
                                   replacement=True, generator=sites_rng)
        chart = torch.stack((pixels.remainder(SIDE),
                             pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
        truth = sample['centre'].flatten(1, 2).gather(1, pixels[..., None].expand(-1, -1, 3))
        panel.append({'donor_id': donor, 'experiment_id': experiment,
                      'source_archive_sha256': expected, 'base': base, 'variant': variant,
                      'seed': seed, 'physical_section_id': sample['provenance'][0]['physical_section_id'],
                      'input': sample['inputs'][0].cpu(), 'chart': chart[0].cpu(),
                      'truth': truth[0].cpu(), 'pixels': pixels[0].cpu(),
                      'offsets': sample['offsets'][0].cpu(), 'weights': sample['weights'][0].cpu()})
        accepted += 1

RUN.mkdir(parents=True)
torch.save(panel, RUN / 'panel.pt')
config = {'seed': SEED, 'cases_per_donor': CASES, 'base_range': [56, 64],
          'source_bindings': bindings, 'approximate_registration_not_physical_oblique_truth': True,
          'atlas_template_sha256': context['provenance']['atlas_binding']['template_sha256'],
          'panel_sha256': hashlib.file_digest((RUN / 'panel.pt').open('rb'), 'sha256').hexdigest(),
          'generator_sha256': hashlib.sha256(Path(__file__).with_name('arbitrary_plane_fov_balance_v10.py').read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'checkpoint_sha256': {name: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                                for name, path in CHECKPOINTS.items()},
          'calibrated': False, 'public_benchmark_used': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

rows = []
flags = torch.arange(2, device='cuda')[None, None]
for name, path in CHECKPOINTS.items():
    model, _ = load_one_shot_checkpoint(path, 'cuda')
    with torch.inference_mode():
        for first_case in range(0, len(panel), 4):
            cases = panel[first_case:first_case + 4]
            inputs = torch.stack([case['input'] for case in cases]).cuda()
            chart = torch.stack([case['chart'] for case in cases]).cuda()
            truth = torch.stack([case['truth'] for case in cases]).cuda()
            offsets = torch.stack([case['offsets'] for case in cases]).cuda()
            weights = torch.stack([case['weights'] for case in cases]).cuda()
            pixels = torch.stack([case['pixels'] for case in cases]).cuda()
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflected = flags.expand(len(cases), model.modes, 2)
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            points = rigid_points_090(states, reflected, chart[:, None, None], (SIDE, SIDE))
            errors = (points - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            first = mass.argmax(-1)
            top16 = mass.topk(16, -1).indices
            mapping = model.map(prediction, offsets, first[:, None] // 2,
                                first[:, None] % 2, (SIDE, SIDE), atlas, weights)
            sites = mapping['centre_surface_ccf_ap_dv_ml_um'][:, 0].flatten(1, 2).gather(
                1, pixels[..., None].expand(-1, -1, 3))
            mapped_error = (sites - truth).norm(dim=-1).mean(-1) / 1000
            for index, case in enumerate(cases):
                rows.append({'model': name, 'donor_id': case['donor_id'],
                             'experiment_id': case['experiment_id'],
                             'source_archive_sha256': case['source_archive_sha256'],
                             'physical_section_id': case['physical_section_id'],
                             'first_rigid_mm': float(errors[index, first[index]]),
                             'top16_oracle_mm': float(errors[index, top16[index]].min()),
                             'first_map_mm': float(mapped_error[index])})
    print(json.dumps({'event': 'model_readout', 'model': name}), flush=True)
    del model
    torch.cuda.empty_cache()

with (RUN / 'rows.jsonl').open('w', encoding='utf8') as file:
    for row in rows:
        file.write(json.dumps(row, allow_nan=False) + '\n')
summary = {}
for name in CHECKPOINTS:
    for donor in (row[2] for row in SOURCES):
        group = [row for row in rows if row['model'] == name and row['donor_id'] == donor]
        summary[f'{name}_{donor}'] = {
            'n': len(group), 'first_rigid_mean_mm': float(np.mean([r['first_rigid_mm'] for r in group])),
            'first_gt5mm': int(np.sum([r['first_rigid_mm'] > 5 for r in group])),
            'top16_capture_le1p5mm': int(np.sum([r['top16_oracle_mm'] <= 1.5 for r in group])),
            'first_map_mean_mm': float(np.mean([r['first_map_mm'] for r in group]))}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
