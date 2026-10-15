"""Matched short TRAIN pilot: individual-brain red anatomy versus one atlas source."""
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
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
from PIL import Image

from training.arbitrary_plane_fov_balance_v10 import sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_red_source_mix_311 import ROOT as RED_ROOT, load_red_source_mix_311
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, UPDATES = 20261015319, 192, 2000
READOUTS = (0, 1000, 2000)
PARENT = ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt'
RUN = ROOT / 'runs/red_source_diversity_319_001'
DEV = ((257636467, 607916488, 182849231), (526501457, 527105701, 517337553))
assert not RUN.exists() and PARENT.drive.upper() == RUN.drive.upper() == 'I:'
with PARENT.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert parent_sha == '7e4be46fe0c8e6637e183865adce9b2ab554058b81b341bdf98bf0fd2ad0d797'

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_red_source_mix_311('cuda')
brain = context['atlas'][1].cpu().numpy() > 0
qc = {row['donor_id']: row for row in json.loads((ROOT /
    'reports/allen_stpt_red_six_donor_qc_310.json').read_text(encoding='utf8'))}
dev_intensities = []
for experiment, file_id, donor in DEV:
    archive = RED_ROOT / f'{experiment}_{file_id}.zip'
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == qc[donor]['archive_sha256']
    with ZipFile(archive) as source:
        red = np.frombuffer(source.read('resampled_red.raw'), dtype='<u2').reshape(
            456, 320, 528).transpose(2, 1, 0)
    low, high = np.quantile(red[brain], (.01, .99))
    normalized = np.clip((red.astype(np.float32) - low) / (high - low), 0, 1)
    normalized[~brain] = 0
    dev_intensities.append((donor, normalized))

RUN.mkdir(parents=True)
config = {'seed': SEED, 'updates_per_arm': UPDATES, 'readout_steps': READOUTS,
    'parent_sha256': parent_sha, 'train_red_sources': context['source_identities'][1:],
    'dev_red_sources': [{'donor_id': donor, 'archive_sha256': qc[donor]['archive_sha256']}
                        for _, _, donor in DEV],
    'train_slots': 'four independent v10 cuts; control all Allen average; treatment last slot cycles three TRAIN red donors',
    'source_geometry': 'registered to CCF; regional TRAIN handedness 18/18 direct-over-ML-flip; weak biological coordinates',
    'readout': 'fresh CCF v10 inner bases 56-63, two DEV red virtual-cut donors, AL1A acquired TRAIN-inner human plane',
    'gate': 'at fixed step 2000: CCF mean first-choice gain >=0.35 mm, no extra >5-mm CCF cases, mean DEV red gain >=0.20 mm, AL1A human-plane mean regression <=0.10 mm',
    'not_evidence_of': 'physical oblique accuracy, exact local warp, calibrated probabilities or GUI readiness'}
(RUN / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')

panel_rng = np.random.default_rng(SEED + 1)
sites_rng = torch.Generator(device='cuda').manual_seed(SEED + 2)
panels = {}
for cohort, volume, count in [('ccf', context['source_intensities'][0], 128),
                              *[(f'red_{donor}', intensity, 64)
                                for donor, intensity in dev_intensities]]:
    context['atlas'][0].copy_(torch.as_tensor(volume, device='cuda'))
    cases = []
    for slot in range(count):
        while True:
            base = 56 + slot % 8
            variant = int(panel_rng.integers(VARIANTS))
            seed = int(panel_rng.integers(2**63 - 1, dtype=np.int64))
            with torch.no_grad():
                sample = sample_fov_balance_v10(context,
                    [base * VARIANTS + variant], seed, SIDE)
            if bool(sample['eligible'][0]):
                break
        valid = sample['valid_mask'][0]
        pixels = torch.multinomial(valid.flatten().float(), 128, replacement=True,
                                  generator=sites_rng)
        chart = torch.stack((pixels.remainder(SIDE),
            pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
        cases.append({'inputs': sample['inputs'][0].cpu(), 'chart': chart.cpu(),
            'truth': sample['centre'][0].reshape(-1, 3)[pixels].cpu(),
            'base': base, 'variant': variant, 'seed': seed,
            'physical_section_id': sample['provenance'][0]['physical_section_id']})
    panels[cohort] = cases
torch.save(panels, RUN / 'panels.pt')
del dev_intensities
context['atlas'][0].copy_(context['source_intensities'][0])

guard_images, guard_ouvs, guard_sections = [], [], []
with ZipFile(ROOT / 'data/lupori_joint_185/rawData.zip') as archive:
    animal = 'AL1A'
    valid = {item.findtext('name') for item in ElementTree.fromstring(
        archive.read(f'{animal}/{animal}-info.xml')).findall('slices')
        if item.findtext('valid') == '1'}
    registration = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))
    for item in registration['slices']:
        section = Path(item['filename']).name.removesuffix('-thumb.png')
        if section not in valid:
            continue
        rgb = np.asarray(Image.open(BytesIO(archive.read(
            f'{animal}/{item["filename"]}'))).convert('RGB').resize(
            (SIDE, SIDE), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        raw_ouv = torch.tensor(item['anchoring'], dtype=torch.float32).reshape(3, 3)
        guard_images.append(gray)
        guard_sections.append(section)
        guard_ouvs.append(torch.stack((quicknii_to_allen_points(raw_ouv[0]) * 25,
            quicknii_to_allen_vectors(raw_ouv[1]) * 25,
            quicknii_to_allen_vectors(raw_ouv[2]) * 25)))
assert len(guard_images) == 52
guard_images = torch.from_numpy(np.stack(guard_images)).cuda()
guard_ouvs = torch.stack(guard_ouvs).cuda()
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
flags = torch.arange(2, device='cuda')[None, None]

with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(PARENT, map_location='cpu', weights_only=True)
assert saved['step'] == 25000 and saved['arm'] == 'single'
schedule, all_readouts = [], {}
for arm in ('control', 'red'):
    model, _ = load_one_shot_checkpoint(PARENT, device='cuda')
    model.requires_grad_(False)
    pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
    shared_modules = (model.encoder, model.lateral)
    for module in (*pose_modules, *shared_modules):
        module.requires_grad_(True)
    groups = [{'params': [p for module in pose_modules for p in module.parameters()], 'lr': 4e-6},
              {'params': [p for module in shared_modules for p in module.parameters()], 'lr': 1e-6}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    optimizer.load_state_dict(saved['optimizer'])
    for group, rate in zip(optimizer.param_groups, (4e-6, 1e-6)):
        group['lr'] = rate
    params = [p for group in groups for p in group['params']]
    rng = np.random.default_rng(SEED + 3)
    site_rng = torch.Generator(device='cuda').manual_seed(SEED + 4)
    torch.manual_seed(SEED + 5)
    torch.cuda.manual_seed_all(SEED + 5)
    readouts, losses = {}, []
    model.train()
    for step in range(UPDATES + 1):
        if step:
            samples, draws = [], []
            for slot in range(4):
                source_index = 1 + (step - 1) % 3 if arm == 'red' and slot == 3 else 0
                context['atlas'][0].copy_(context['source_intensities'][source_index])
                while True:
                    if arm == 'control':
                        base = int(rng.integers(56))
                        variant = int(rng.integers(VARIANTS))
                        seed = int(rng.integers(2**63 - 1, dtype=np.int64))
                    else:
                        base, variant, seed = schedule[step - 1][slot]
                    with torch.no_grad():
                        sample = sample_fov_balance_v10(context,
                            [base * VARIANTS + variant], seed, SIDE)
                    if bool(sample['eligible'][0]):
                        break
                    assert arm == 'control'
                samples.append(sample)
                draws.append((base, variant, seed))
            if arm == 'control':
                schedule.append(draws)
            inputs = torch.cat([sample['inputs'] for sample in samples])
            state = torch.cat([sample['state'] for sample in samples])
            reflection = torch.cat([sample['reflection'] for sample in samples])
            valid = torch.cat([sample['valid_mask'] for sample in samples])
            centres = torch.cat([sample['centre'] for sample in samples])
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(4, model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            pixels = torch.multinomial(valid.flatten(1).float(), 128,
                replacement=True, generator=site_rng)
            chart = torch.stack((pixels.remainder(SIDE),
                pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            truth = centres.flatten(1, 2).gather(1, pixels[..., None].expand(-1, -1, 3))
            proposed = rigid_points_090(states, reflections,
                chart[:, None, None], (SIDE, SIDE))
            five = rigid_points_090(states, reflections, corners, (SIDE, SIDE))
            true_five = rigid_points_090(state, reflection, corners, (SIDE, SIDE))
            tissue_um = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
            five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            true_normal = full_frame_state_to_components(state)[1][..., :, 2]
            normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
            physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-physical.detach() / .7, -1)
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(4, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
                + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
                + .5 * physical[:, :2 * model.base_modes].min(-1).values
                + (neighbourhood * near_cost).sum(-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        if step in READOUTS:
            model.eval()
            rows = []
            with torch.inference_mode():
                for cohort, cases in panels.items():
                    for start in range(0, len(cases), 4):
                        group = cases[start:start + 4]
                        batch = torch.stack([case['inputs'] for case in group]).cuda()
                        chart = torch.stack([case['chart'] for case in group]).cuda()
                        truth = torch.stack([case['truth'] for case in group]).cuda()
                        prediction = model.predict(batch)
                        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                        reflections = flags.expand(len(group), model.modes, 2)
                        mass = (prediction['log_mass'][..., None] + torch.stack((
                            F.logsigmoid(-prediction['reflection_logit']),
                            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                        mapped = rigid_points_090(states, reflections,
                            chart[:, None, None], (SIDE, SIDE))
                        errors = (mapped - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                        first = mass.argmax(-1)
                        top16 = mass.topk(16, -1).indices
                        for i, case in enumerate(group):
                            rows.append({'cohort': cohort, 'base': case['base'],
                                'physical_section_id': case['physical_section_id'],
                                'first_mm': float(errors[i, first[i]]),
                                'top16_oracle_mm': float(errors[i, top16[i]].min())})
                for start in range(0, 52, 4):
                    ids = list(range(start, min(start + 4, 52)))
                    batch = torch.zeros(len(ids), 5, SIDE, SIDE, device='cuda')
                    batch[:, 0] = guard_images[ids]
                    prediction = model.predict(batch)
                    states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                    reflections = flags.expand(len(ids), model.modes, 2)
                    mass = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                    mapped = rigid_points_090(states, reflections, corners, (SIDE, SIDE))
                    reference = guard_ouvs[ids, 0, None] + corners[None, :, :1] * guard_ouvs[ids, 1, None] \
                        + corners[None, :, 1:] * guard_ouvs[ids, 2, None]
                    errors = (mapped - reference[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                    first = mass.argmax(-1)
                    rows.extend({'cohort': 'AL1A', 'section_id': guard_sections[ids[i]],
                                 'first_mm': float(errors[i, first[i]])}
                        for i in range(len(ids)))
            metrics = {cohort: {'n': sum(row['cohort'] == cohort for row in rows),
                'first_mean_mm': float(np.mean([row['first_mm'] for row in rows if row['cohort'] == cohort])),
                'first_gt5': sum(row['first_mm'] > 5 for row in rows if row['cohort'] == cohort)}
                for cohort in ('ccf', *(f'red_{donor}' for _, _, donor in DEV), 'AL1A')}
            readouts[str(step)] = {'metrics': metrics, 'rows': rows,
                'mean_train_loss_since_last_readout': float(np.mean(losses)) if losses else None}
            (RUN / f'{arm}_{step:04d}_readout.json').write_text(
                json.dumps(readouts[str(step)]) + '\n', encoding='utf8')
            if step:
                torch.save({'step': 25000 + step, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'config': config,
                    'arm': arm, 'calibrated': False}, RUN / f'{arm}_{step:04d}.pt')
            print(json.dumps({'arm': arm, 'step': step, 'metrics': metrics,
                'train_loss': readouts[str(step)]['mean_train_loss_since_last_readout']}), flush=True)
            losses.clear()
            model.train()
    all_readouts[arm] = readouts
    del model, optimizer
    torch.cuda.empty_cache()

(RUN / 'schedule.json').write_text(json.dumps(schedule) + '\n', encoding='utf8')
control, red = (all_readouts[arm][str(UPDATES)]['metrics'] for arm in ('control', 'red'))
red_control = np.mean([control[f'red_{donor}']['first_mean_mm'] for _, _, donor in DEV])
red_treatment = np.mean([red[f'red_{donor}']['first_mean_mm'] for _, _, donor in DEV])
gate = {'ccf_gain_ge0p35': bool(red['ccf']['first_mean_mm'] <= control['ccf']['first_mean_mm'] - .35),
        'ccf_gt5_not_worse': bool(red['ccf']['first_gt5'] <= control['ccf']['first_gt5']),
        'dev_red_gain_ge0p20': bool(red_treatment <= red_control - .20),
        'AL1A_regression_le0p10': bool(red['AL1A']['first_mean_mm'] <= control['AL1A']['first_mean_mm'] + .10)}
(RUN / 'decision.json').write_text(json.dumps({'gate': gate, 'pass': all(gate.values()),
    'control': control, 'red': red}, indent=2) + '\n', encoding='utf8')
print(json.dumps({'event': 'decision', 'gate': gate, 'pass': all(gate.values())}), flush=True)
