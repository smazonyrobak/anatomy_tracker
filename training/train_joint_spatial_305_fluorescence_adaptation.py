"""Matched 305 scratch-lineage fluorescence adaptation; TRAIN animals only."""
import hashlib
import json
import math
import os
import sys
import time
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, UPDATES = 20261014305, 192, 600
STEPS = (0, 200, 400, 600)
TRAIN = ('CC2B', 'CC4B', 'CC4A')
GUARD = 'AL1A'
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/JOINT_SPATIAL_305_FLUORESCENCE_ADAPTATION_PROTOCOL_20261014.md'
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
PARENT = ROOT / 'runs/joint_spatial_feedback_305_pose_002/pose_step_26000.pt'
PANEL = PARENT.parent / 'inner_panel.pt'
RUN = ROOT / 'runs/joint_spatial_305_fluorescence_adaptation_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


split = json.loads(SPLIT.read_text(encoding='utf8'))
assert tuple(split['train']) == (GUARD, *TRAIN)
assert ARCHIVE.stat().st_size == split['archive_bytes']
parent_receipt = json.loads((PARENT.parent / 'completed.json').read_text(encoding='utf8'))
assert parent_receipt['summary']['selected_step'] == 26000
assert parent_receipt['output_sha256'][PARENT.name] == sha(PARENT)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64('cuda')
panel = torch.load(PANEL, map_location='cpu', weights_only=False)
assert len(panel) == 64 and {row['base'] for row in panel} == set(range(56, 64))

records, images, masks, ouvs = [], [], [], []
with ZipFile(ARCHIVE) as archive:
    for animal in (GUARD, *TRAIN):
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            archive.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        registration = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))
        assert registration['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        assert registration['target-resolution'] == [456, 528, 320]
        for item in registration['slices']:
            section = Path(item['filename']).name.removesuffix('-thumb.png')
            if section not in valid:
                continue
            member = f'{animal}/{item["filename"]}'
            raw = archive.read(member)
            rgb = np.asarray(Image.open(BytesIO(raw)).convert('RGB').resize(
                (SIDE, SIDE), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
            mask = np.asarray(Image.open(BytesIO(archive.read(
                f'{animal}/masks/{section}-mask.png'))).convert('L').resize(
                    (SIDE, SIDE), Image.Resampling.NEAREST)) > 127
            source_ouv = torch.tensor(item['anchoring'], dtype=torch.float32).reshape(3, 3)
            o = quicknii_to_allen_points(source_ouv[0]) * 25
            u = quicknii_to_allen_vectors(source_ouv[1]) * 25
            v = quicknii_to_allen_vectors(source_ouv[2]) * 25
            records.append({'animal_id': animal, 'specimen_id': animal,
                'experiment_id': None, 'source_section_id': section,
                'archive_member': member, 'source_image_sha256': hashlib.sha256(raw).hexdigest()})
            images.append(gray)
            masks.append(mask)
            ouvs.append(torch.stack((o, u, v)))
assert len(records) == 240 and len({(r['animal_id'], r['source_section_id'])
                                    for r in records}) == 240
guard_indices = [i for i, row in enumerate(records) if row['animal_id'] == GUARD]
train_indices = [i for i, row in enumerate(records) if row['animal_id'] in TRAIN]
assert len(guard_indices) == 52 and len(train_indices) == 188
images = torch.from_numpy(np.stack(images)).to('cuda')
masks = torch.from_numpy(np.stack(masks)).to('cuda')
ouvs = torch.stack(ouvs).to('cuda')
normal_targets = F.normalize(torch.cross(ouvs[:, 1], ouvs[:, 2], dim=-1), dim=-1)

config = {'candidate': 305, 'seed': SEED, 'optimizer_batches': UPDATES,
    'readouts': STEPS, 'synthetic_per_batch': 2, 'real_per_batch': 1,
    'synthetic_train_bases': list(range(56)), 'synthetic_inner_bases': list(range(56, 64)),
    'gradient_animals': TRAIN, 'inner_guard_animal': GUARD,
    'gradient_real_sections': len(train_indices), 'guard_sections': len(guard_indices),
    'source_doi': split['source_doi'], 'archive_bytes': split['archive_bytes'],
    'archive_md5_from_verified_source': split['archive_md5'],
    'input': 'max(red,green)/255, image-wide p99, four zero channels at 192 pixels',
    'model_start': '305 project-random-initialized scratch pose step 26000; no earlier model lineage',
    'matched_arms': {'control_real_loss_weight': 0., 'adapt_real_loss_weight': 1.},
    'optimized_modules': ['encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global'],
    'optimizer': 'fresh matched AdamW; pose 2e-5, shared 5e-6; cosine to 20%; weight decay 1e-4',
    'label_role': 'approximate human VisuAlign plane, weak TRAIN supervision only',
    'parent_checkpoint_sha256': sha(PARENT), 'parent_completion_sha256': sha(PARENT.parent / 'completed.json'),
    'panel_sha256': sha(PANEL), 'protocol_sha256': sha(PROTOCOL),
    'split_sha256': sha(SPLIT), 'source_sha256': sha(SOURCE),
    'no_external_weights_features_or_pseudolabels': True,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
flags = torch.arange(2, device='cuda')[None, None]
all_readouts = {}

for arm in ('control', 'adapt'):
    path = RUN / arm
    path.mkdir()
    model, _ = load_one_shot_checkpoint(PARENT, device='cuda')
    model.requires_grad_(False)
    pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
    shared_modules = (model.encoder, model.lateral)
    for module in (*pose_modules, *shared_modules):
        module.requires_grad_(True)
    groups = [
        {'params': [p for module in pose_modules for p in module.parameters()], 'base_lr': 2e-5},
        {'params': [p for module in shared_modules for p in module.parameters()], 'base_lr': 5e-6},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    parameters = [p for group in groups for p in group['params']]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(train_indices)
    cursor = epoch = attempts = 0
    started = time.perf_counter()
    readouts = {}
    with (path / 'draws.jsonl').open('w', encoding='utf8') as draws, \
            (path / 'training.jsonl').open('w', encoding='utf8') as training, \
            (path / 'readouts.jsonl').open('w', encoding='utf8') as readout_file:
        for step in range(UPDATES + 1):
            if step:
                samples = []
                for slot in range(2):
                    while True:
                        attempts += 1
                        base = int(rng.integers(56))
                        variant = int(rng.integers(VARIANTS))
                        seed = int(rng.integers(2**63 - 1, dtype=np.int64))
                        with torch.no_grad():
                            sample = sample_postartifact_gauge_v1(
                                context, [base * VARIANTS + variant], seed, SIDE)
                        used = bool(sample['eligible'][0])
                        draws.write(json.dumps({'kind': 'synthetic', 'step': step,
                            'slot': slot, 'attempt': attempts, 'base': base,
                            'variant': variant, 'seed': seed, 'accepted': used,
                            'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                        if used:
                            samples.append(sample)
                            break
                if cursor == len(order):
                    order = rng.permutation(train_indices)
                    cursor = 0
                    epoch += 1
                index = int(order[cursor])
                cursor += 1
                draws.write(json.dumps({'kind': 'real', 'step': step, 'slot': 2,
                    'epoch': epoch, **records[index]}, allow_nan=False) + '\n')
                synthetic = torch.cat([s['inputs'] for s in samples])
                target_state = torch.cat([s['state'] for s in samples])
                target_reflection = torch.cat([s['reflection'] for s in samples])
                valid = torch.cat([s['valid_mask'] for s in samples])
                centres = torch.cat([s['centre'] for s in samples])
                real = torch.zeros(1, 5, SIDE, SIDE, device='cuda')
                real[0, 0] = images[index]
                inputs = torch.cat((synthetic, real))
                prediction = model.predict(inputs)
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflections = flags.expand(len(states), model.modes, 2)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)

                pixels = torch.multinomial(valid.flatten(1).float(), 128, replacement=True)
                chart = torch.stack((pixels.remainder(SIDE),
                    pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
                truth = centres.flatten(1, 2).gather(1, pixels[..., None].expand(-1, -1, 3))
                proposed = rigid_points_090(states[:2], reflections[:2],
                                            chart[:, None, None], (SIDE, SIDE))
                five = rigid_points_090(states[:2], reflections[:2], corners, (SIDE, SIDE))
                true_five = rigid_points_090(target_state, target_reflection,
                                             corners, (SIDE, SIDE))
                tissue_um = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
                five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
                normal = full_frame_state_to_components(prediction['state'][:2])[1][..., :, 2]
                true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
                normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
                physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
                target_mass = F.softmax(-physical.detach() / .7, -1)
                nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
                near_mode = model.base_modes + nearest.indices
                near_cost = physical.reshape(2, model.modes, 2).min(-1).values.gather(1, near_mode)
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
                synthetic_loss = (-1.5 * torch.logsumexp(prior[:2] - physical / 1.5, -1)
                    + .5 * F.kl_div(prior[:2], target_mass, reduction='none').sum(-1)
                    + .5 * physical[:, :2 * model.base_modes].min(-1).values
                    + (neighbourhood * near_cost).sum(-1)).mean()

                rpixels = torch.multinomial(masks[index].flatten().float(), 128, replacement=True)
                rchart = torch.stack((rpixels.remainder(SIDE),
                    rpixels.div(SIDE, rounding_mode='floor')), -1).float()[None] / SIDE
                reference = ouvs[index, 0] + rchart[..., :1] * ouvs[index, 1] + \
                    rchart[..., 1:] * ouvs[index, 2]
                rproposed = rigid_points_090(states[2:3], reflections[2:3],
                                              rchart[:, None, None], (SIDE, SIDE))
                reference_five = ouvs[index, 0] + corners[:, :1] * ouvs[index, 1] + \
                    corners[:, 1:] * ouvs[index, 2]
                rfive = rigid_points_090(states[2:3], reflections[2:3], corners, (SIDE, SIDE))
                tissue_um = (rproposed - reference[:, None, None]).norm(dim=-1).mean(-1)
                five_um = (rfive - reference_five).norm(dim=-1).mean(-1)
                rnormal = full_frame_state_to_components(prediction['state'][2:3])[1][..., :, 2]
                normal_um = 4000 * (1 - (rnormal[0] * normal_targets[index]).sum(-1
                    ).abs().clamp_max(1))
                real_cost = (.75 * tissue_um + .25 * five_um + normal_um[None, :, None]
                             ).flatten(1) / 1000
                real_mass = F.softmax(-real_cost.detach() / .7, -1)
                nearest = (normal_targets[index] @ model.normal_anchor_frames[:, :, 2].T
                           ).abs().topk(4)
                near_mode = model.base_modes + nearest.indices
                near_cost = real_cost.reshape(model.modes, 2).min(-1).values[near_mode]
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:1]), -1)
                real_loss = (-1.5 * torch.logsumexp(prior[2] - real_cost[0] / 1.5, -1)
                    + .5 * F.kl_div(prior[2], real_mass[0], reduction='sum')
                    + .5 * real_cost[0, :2 * model.base_modes].min()
                    + (neighbourhood * near_cost).sum())
                loss = synthetic_loss + (real_loss if arm == 'adapt' else 0 * real_loss)
                decay = .2 + .4 * (1 + math.cos(math.pi * step / UPDATES))
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * decay
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                training.write(json.dumps({'step': step, 'synthetic_presentations': 2 * step,
                    'real_presentations': step, 'real_epoch': epoch,
                    'synthetic_loss': float(synthetic_loss.detach()),
                    'real_weak_loss': float(real_loss.detach()), 'loss': float(loss.detach()),
                    'gradient_norm': float(gradient),
                    'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
                if step in STEPS:
                    training.flush()
                    draws.flush()
                    print(json.dumps({'event': 'train_milestone', 'arm': arm, 'step': step,
                        'synthetic_loss': float(synthetic_loss.detach()),
                        'real_weak_loss': float(real_loss.detach())}), flush=True)

            if step in STEPS:
                model.eval()
                rows = []
                with torch.inference_mode():
                    for cohort in ('guard', 'synthetic'):
                        count = len(guard_indices) if cohort == 'guard' else len(panel)
                        for start in range(0, count, 4):
                            if cohort == 'guard':
                                ids = guard_indices[start:start + 4]
                                batch = torch.zeros(len(ids), 5, SIDE, SIDE, device='cuda')
                                batch[:, 0] = images[ids]
                                chart = corners[None].expand(len(ids), -1, -1)
                                reference = ouvs[ids, 0, None] + chart[..., :1] * ouvs[ids, 1, None] \
                                    + chart[..., 1:] * ouvs[ids, 2, None]
                                true_normals = normal_targets[ids]
                                identities = [records[i] for i in ids]
                            else:
                                group = panel[start:start + 4]
                                batch = torch.stack([row['inputs'] for row in group]).cuda()
                                chart = torch.stack([row['chart'] for row in group]).cuda()
                                reference = torch.stack([row['truth'] for row in group]).cuda()
                                true_states = torch.stack([row['state'] for row in group]).cuda()
                                true_normals = full_frame_state_to_components(true_states)[1][..., :, 2]
                                identities = [{'physical_section_id': row['physical_section_id'],
                                    'base': row['base'], 'obliquity_deg': row['obliquity_deg'],
                                    'visible_fraction': row['visible_fraction'],
                                    'exterior_style': row['exterior_style']} for row in group]
                            p = model.predict(batch)
                            s = p['state'][:, :, None].expand(-1, -1, 2, -1)
                            r = flags.expand(len(s), model.modes, 2)
                            mass = (p['log_mass'][..., None] + torch.stack((
                                F.logsigmoid(-p['reflection_logit']),
                                F.logsigmoid(p['reflection_logit'])), -1)).flatten(1)
                            mapped = rigid_points_090(s, r, chart[:, None, None], (SIDE, SIDE))
                            errors = (mapped - reference[:, None, None]).norm(dim=-1
                                ).mean(-1).flatten(1) / 1000
                            first = mass.argmax(-1)
                            top16 = mass.topk(16, -1).indices
                            pred_normal = full_frame_state_to_components(p['state'])[1][..., :, 2]
                            angle = torch.rad2deg(torch.acos((pred_normal * true_normals[:, None]
                                ).sum(-1).abs().clamp_max(1)))
                            for j, identity in enumerate(identities):
                                rows.append({'cohort': cohort, **identity,
                                    'first_five_or_site_gap_mm': float(errors[j, first[j]]),
                                    'top16_oracle_gap_mm': float(errors[j, top16[j]].min()),
                                    'first_unsigned_normal_gap_deg': float(angle[j, first[j] // 2])})
                metrics = {}
                for cohort in ('guard', 'synthetic'):
                    group = [row for row in rows if row['cohort'] == cohort]
                    metrics[cohort] = {'n': len(group),
                        'mean_first_gap_mm': float(np.mean([r['first_five_or_site_gap_mm'] for r in group])),
                        'mean_top16_oracle_gap_mm': float(np.mean([r['top16_oracle_gap_mm'] for r in group])),
                        'top16_capture_1p5mm': sum(r['top16_oracle_gap_mm'] <= 1.5 for r in group),
                        'median_first_unsigned_normal_gap_deg': float(np.median(
                            [r['first_unsigned_normal_gap_deg'] for r in group]))}
                readouts[str(step)] = metrics
                readout_file.write(json.dumps({'step': step, 'metrics': metrics,
                    'rows': rows}, allow_nan=False) + '\n')
                readout_file.flush()
                torch.save({'step': step, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'config': config, 'arm': arm,
                    'calibrated': False}, path / f'pose_step_{step:05d}.pt')
                print(json.dumps({'event': 'readout', 'arm': arm, 'step': step,
                    'metrics': metrics}), flush=True)
                model.train()
    all_readouts[arm] = readouts
    (path / 'completed.json').write_text(json.dumps({'arm': arm, 'updates': UPDATES,
        'readouts': readouts, 'output_sha256': {file.name: sha(file) for file in path.iterdir()
            if file.name != 'completed.json'}, 'calibrated': False}, indent=2), encoding='utf8')
    del model, optimizer
    torch.cuda.empty_cache()

assert sha(RUN / 'control/draws.jsonl') == sha(RUN / 'adapt/draws.jsonl')
c = all_readouts['control']['600']
a = all_readouts['adapt']['600']
reduction = c['guard']['mean_first_gap_mm'] - a['guard']['mean_first_gap_mm']
gate = (reduction >= 1.0 and reduction / c['guard']['mean_first_gap_mm'] >= .25
    and a['synthetic']['mean_first_gap_mm'] <= c['synthetic']['mean_first_gap_mm'] + .2
    and a['synthetic']['top16_capture_1p5mm'] >= c['synthetic']['top16_capture_1p5mm'] - 5)
summary = {'step_600_gate_pass': gate, 'guard_absolute_gain_mm': reduction,
    'guard_relative_gain': reduction / c['guard']['mean_first_gap_mm'],
    'readouts': all_readouts, 'matched_draws_sha256': sha(RUN / 'control/draws.jsonl'),
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(RUN / 'completed.json').write_text(json.dumps({'summary': summary,
    'output_sha256': {'config.json': sha(RUN / 'config.json'),
        'summary.json': sha(RUN / 'summary.json'),
        'control/completed.json': sha(RUN / 'control/completed.json'),
        'adapt/completed.json': sha(RUN / 'adapt/completed.json')}}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'gate_pass': gate,
                  'guard_absolute_gain_mm': reduction}), flush=True)
