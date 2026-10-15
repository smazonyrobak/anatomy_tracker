"""Bounded, from-scratch C4 joint pose/tissue-map study; TRAIN data only."""
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

from training.arbitrary_plane_c4_joint_model import C4JointSliceModel
from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

SEED, SIDE, UPDATES = 20261015321, 192, 12000
STAGES = (0, 1000, 4000, 12000)
TRAIN, GUARD = ('CC2B', 'CC4B', 'CC4A'), 'AL1A'
RUN = ROOT / 'runs/c4_joint_321_002'
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
SPLIT = Path(__file__).with_name('lupori_animal_split_185.json')
COMPARATOR = ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt'
SOURCE = Path(__file__).resolve()
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
assert json.loads(SPLIT.read_text(encoding='utf8'))['train'] == [GUARD, *TRAIN]
with COMPARATOR.open('rb') as stream:
    comparator_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert comparator_sha == '7e4be46fe0c8e6637e183865adce9b2ab554058b81b341bdf98bf0fd2ad0d797'

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_fov_balance_v10('cuda')
assert len(context['bases']) == 64 and len(context['subjects']) == 64 * VARIANTS

records, images, ouvs = [], [], []
with ZipFile(ARCHIVE) as archive:
    for animal in (GUARD, *TRAIN):
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            archive.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        registration = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))
        assert registration['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
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
            anchoring = torch.as_tensor(item['anchoring'], dtype=torch.float32).reshape(3, 3)
            records.append({'animal_id': animal, 'specimen_id': animal,
                'experiment_id': None, 'source_section_id': section,
                'archive_member': member, 'source_image_sha256': hashlib.sha256(raw).hexdigest()})
            images.append(gray)
            ouvs.append(torch.stack((quicknii_to_allen_points(anchoring[0]) * 25,
                quicknii_to_allen_vectors(anchoring[1]) * 25,
                quicknii_to_allen_vectors(anchoring[2]) * 25)))
assert len(records) == 240
images = torch.from_numpy(np.stack(images)).cuda()
ouvs = torch.stack(ouvs).cuda()
guard_ids = [i for i, row in enumerate(records) if row['animal_id'] == GUARD]
train_ids = {animal: [i for i, row in enumerate(records) if row['animal_id'] == animal]
             for animal in TRAIN}
assert len(guard_ids) == 52 and sum(map(len, train_ids.values())) == 188
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE


def rigid_candidates(model, prediction, chart):
    batch, modes = prediction['state'].shape[:2]
    branch = torch.arange(modes, device=chart.device)[None, :, None].expand(batch, -1, 2)
    reflection = torch.arange(2, device=chart.device)[None, None].expand(batch, modes, -1)
    local_chart = model.canonical_chart(
        chart[:, None, None].expand(-1, modes, 2, -1, -1),
        branch // model.normal_modes, SIDE)
    states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
    return rigid_points_090(states, reflection, local_chart, (SIDE, SIDE))


def branch_scores(prediction):
    reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                              F.logsigmoid(prediction['reflection_logit'])), -1)
    return (prediction['log_mass'][..., None] + reflection).flatten(1)


RUN.mkdir(parents=True, exist_ok=False)
panel_rng = np.random.default_rng(SEED + 1)
panel_sites = torch.Generator(device='cuda').manual_seed(SEED + 2)
panels = {}
with (RUN / 'panel_draws.jsonl').open('w', encoding='utf8') as draws, torch.no_grad():
    for cohort, count in (('inner', 128), ('steep', 32), ('sparse', 32)):
        cases = []
        for slot in range(count):
            while True:
                base = int(panel_rng.integers(56, 64))
                variant = int(panel_rng.integers(VARIANTS))
                seed = int(panel_rng.integers(2**63 - 1, dtype=np.int64))
                sample = sample_fov_balance_v10(context,
                    [base * VARIANTS + variant], seed, SIDE)
                valid = sample['valid_mask'][0]
                visible = float(((sample['visible'][0] > .05) &
                    ((sample['inputs'][0, 2] < .5) | sample['brush_mask'][0])).float().mean())
                normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                obliquity = math.degrees(math.acos(float(normal.abs().max().clamp(0, 1))))
                accepted = bool(sample['eligible'][0]) and (
                    cohort != 'steep' or obliquity >= 45) and (
                    cohort != 'sparse' or visible < .15)
                provenance = sample['provenance'][0]
                draws.write(json.dumps({'cohort': cohort, 'slot': slot, 'base': base,
                    'variant': variant, 'seed': seed, 'accepted': accepted,
                    'visible_fraction': visible, 'obliquity_deg': obliquity,
                    'provenance': provenance}, allow_nan=False) + '\n')
                if accepted:
                    break
            assert provenance['split'] == 'train'
            pixel = torch.multinomial(valid.flatten().float(), 128, replacement=True,
                                      generator=panel_sites)
            chart = torch.stack((pixel.remainder(SIDE),
                pixel.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            cases.append({'input': sample['inputs'][0].cpu(), 'valid': valid.cpu(), 'chart': chart.cpu(),
                'truth': sample['centre'][0].reshape(-1, 3)[pixel].cpu(),
                'meta': {'cohort': cohort, 'base': base, 'variant': variant, 'seed': seed,
                    'physical_section_id': provenance['physical_section_id'],
                    'virtual_subject_id': provenance['virtual_subject_id'],
                    'sampling_branch': provenance['sampling_branch'],
                    'framing': provenance['framing']['kind'],
                    'visible_fraction': visible, 'obliquity_deg': obliquity}})
        panels[cohort] = cases
torch.save(panels, RUN / 'panels.pt')

config = {'candidate': 321, 'seed': SEED, 'optimizer_batches': UPDATES,
    'readout_batches': STAGES, 'synthetic_per_batch': 4,
    'synthetic_gradient_bases': list(range(56)),
    'synthetic_inner_readout_bases': list(range(56, 64)),
    'lupori_gradient_animals': TRAIN, 'lupori_guard_animal': GUARD,
    'acquired_weak_every_n_batches': 4,
    'model_initialization': 'random; no previous weights, features, or pseudo-labels',
    'synthetic_sampler': 'one independent v10 physical section per slot; no appearance pairing or plane de-duplication',
    'pose_loss': 'soft physical branch cost + detached-cost rank target + nearest-normal anchor exposure',
    'map_loss': 'true-state local field + predicted near/first branch maps; pose gradient from predicted branches',
    'weak_real_loss': 'TRAIN Lupori five-site OUV gap with 1mm tolerance and unsigned normal with 100um tolerance',
    'readout': 'generator-coordinate synthetic truth on fresh TRAIN-inner CCF sections; AL1A weak human VisuAlign agreement is a safety guard, not physical truth',
    'checkpoint_selection': 'synthetic inner/steep/sparse first-mapped gap only; AL1A enters predeclared safety gates but not best-score selection',
    'stage_gates': {
        '1000': 'inner first-mapped gain >=0.35mm from random init; inner >5mm tail +<=2; steep/sparse mean +<=0.30mm; AL1A mean +<=0.50mm',
        '4000': 'inner first-mapped gain >=0.15mm from batch1000; inner >5mm tail +<=2; steep/sparse mean +<=0.30mm; AL1A mean +<=0.30mm; mapped no worse than rigid +0.10mm',
        '12000': 'final science gate: inner first mapped better than frozen309 rigid by >=0.35mm and own rigid by >=0.05mm; inner first rigid better by >=0.15mm; >5mm inner tail no worse; steep/sparse no worse by >0.20mm versus frozen309 rigid; AL1A no worse by >0.20mm versus frozen309 weak OUV'},
    'model_source_sha256': hashlib.sha256(SOURCE.with_name(
        'arbitrary_plane_c4_joint_model.py').read_bytes()).hexdigest(),
    'runner_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'v10_source_sha256': hashlib.sha256(SOURCE.with_name(
        'arbitrary_plane_fov_balance_v10.py').read_bytes()).hexdigest(),
    'sampler_input_sha256': context['provenance']['input_sha256'],
    'lupori_split_sha256': hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
    'lupori_archive_bytes': ARCHIVE.stat().st_size,
    'comparator_checkpoint_sha256': comparator_sha,
    'panel_draws_sha256': hashlib.sha256((RUN / 'panel_draws.jsonl').read_bytes()).hexdigest(),
    'panels_sha256': hashlib.file_digest((RUN / 'panels.pt').open('rb'), 'sha256').hexdigest(),
    'no_dev_cal_final_or_deepslice_gradients': True, 'calibrated': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')


def evaluate(model):
    rows = []
    with torch.inference_mode():
        for cohort, cases in panels.items():
            for start in range(0, len(cases), 4):
                group = cases[start:start + 4]
                image = torch.stack([case['input'] for case in group]).cuda()
                chart = torch.stack([case['chart'] for case in group]).cuda()
                truth = torch.stack([case['truth'] for case in group]).cuda()
                prediction = model.predict(image)
                scores = branch_scores(prediction)
                first = scores.argmax(-1)
                errors = (rigid_candidates(model, prediction, chart) -
                          truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
                first_map = model.map_sites(prediction, chart,
                    (first // 2)[:, None], (first % 2)[:, None])
                mapped = (first_map['coordinates'][:, 0] - truth).norm(dim=-1).mean(-1) / 1000
                oracle = errors.gather(1, scores.topk(16, -1).indices).min(-1).values
                valid = torch.stack([case['valid'] for case in group]).cuda()
                field = first_map['local_displacement_um'][:, 0]
                field_rms = ((field.square().sum(1) * valid).sum((1, 2)) /
                             valid.sum((1, 2))).sqrt() / 1000
                basis = full_frame_state_to_components(first_map['state'][:, 0])[2]
                dx = (field[..., 1:] - field[..., :-1]).norm(dim=1) / \
                     (basis[..., 0].norm(dim=-1) / SIDE)[:, None, None]
                dy = (field[:, :, 1:] - field[:, :, :-1]).norm(dim=1) / \
                     (basis[..., 1].norm(dim=-1) / SIDE)[:, None, None]
                for i, case in enumerate(group):
                    strain = torch.cat((dx[i][valid[i, :, 1:] & valid[i, :, :-1]],
                                        dy[i][valid[i, 1:] & valid[i, :-1]]))
                    rows.append({**case['meta'], 'first_rigid_gap_mm': float(errors[i, first[i]]),
                        'first_mapped_gap_mm': float(mapped[i]), 'top16_rigid_oracle_mm': float(oracle[i]),
                        'first_local_rms_mm': float(field_rms[i]),
                        'first_local_gradient_p95': float(torch.quantile(strain, .95))})
        for start in range(0, len(guard_ids), 4):
            ids = guard_ids[start:start + 4]
            image = torch.zeros(len(ids), 5, SIDE, SIDE, device='cuda')
            image[:, 0] = images[ids]
            prediction = model.predict(image)
            first = branch_scores(prediction).argmax(-1)
            reference = ouvs[ids, 0, None] + corners[None, :, :1] * ouvs[ids, 1, None] \
                + corners[None, :, 1:] * ouvs[ids, 2, None]
            predicted = rigid_candidates(model, prediction, corners[None].expand(len(ids), -1, -1))
            errors = (predicted - reference[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            weak_normals = F.normalize(torch.cross(ouvs[ids, 1], ouvs[ids, 2], dim=-1), dim=-1)
            for i, index in enumerate(ids):
                mode = int(first[i]) // 2
                rows.append({'cohort': 'lupori_guard', **records[index],
                    'first_weak_five_site_gap_mm': float(errors[i, first[i]]),
                    'first_axis_correct': int(normals[i, mode].abs().argmax()) ==
                        int(weak_normals[i].abs().argmax())})
    metrics = {}
    for cohort in panels:
        group = [row for row in rows if row['cohort'] == cohort]
        metrics[cohort] = {'n': len(group),
            'mean_first_rigid_gap_mm': float(np.mean([r['first_rigid_gap_mm'] for r in group])),
            'mean_first_mapped_gap_mm': float(np.mean([r['first_mapped_gap_mm'] for r in group])),
            'mean_first_local_rms_mm': float(np.mean([r['first_local_rms_mm'] for r in group])),
            'median_first_local_gradient_p95': float(np.median([
                r['first_local_gradient_p95'] for r in group])),
            'mapped_better_count': sum(r['first_mapped_gap_mm'] < r['first_rigid_gap_mm'] for r in group),
            'first_mapped_over_5mm': sum(r['first_mapped_gap_mm'] > 5 for r in group),
            'top16_rigid_capture_1p5mm': sum(r['top16_rigid_oracle_mm'] <= 1.5 for r in group)}
    guard = [row for row in rows if row['cohort'] == 'lupori_guard']
    metrics['lupori_guard'] = {'n': len(guard),
        'mean_first_weak_five_site_gap_mm': float(np.mean([
            r['first_weak_five_site_gap_mm'] for r in guard])),
        'first_axis_correct': sum(r['first_axis_correct'] for r in guard)}
    assert [metrics[key]['n'] for key in ('inner', 'steep', 'sparse', 'lupori_guard')] == [128, 32, 32, 52]
    return metrics, rows


with torch.inference_mode():
    old, _ = load_one_shot_checkpoint(COMPARATOR, device='cuda')
    comparator_rows = []
    for cohort, cases in panels.items():
        for start in range(0, len(cases), 4):
            group = cases[start:start + 4]
            image = torch.stack([case['input'] for case in group]).cuda()
            chart = torch.stack([case['chart'] for case in group]).cuda()
            truth = torch.stack([case['truth'] for case in group]).cuda()
            prediction = old.predict(image)
            scores = branch_scores(prediction)
            first = scores.argmax(-1)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            flags = torch.arange(2, device='cuda')[None, None].expand(len(group), old.modes, -1)
            mapped = rigid_points_090(states, flags, chart[:, None, None], (SIDE, SIDE))
            errors = (mapped - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            for i, case in enumerate(group):
                comparator_rows.append({**case['meta'],
                    'first_rigid_gap_mm': float(errors[i, first[i]]),
                    'top16_rigid_oracle_mm': float(errors[i, scores[i].topk(16).indices].min())})
    for start in range(0, len(guard_ids), 4):
        ids = guard_ids[start:start + 4]
        image = torch.zeros(len(ids), 5, SIDE, SIDE, device='cuda')
        image[:, 0] = images[ids]
        prediction = old.predict(image)
        first = branch_scores(prediction).argmax(-1)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        flags = torch.arange(2, device='cuda')[None, None].expand(len(ids), old.modes, -1)
        predicted = rigid_points_090(states, flags, corners, (SIDE, SIDE))
        reference = ouvs[ids, 0, None] + corners[None, :, :1] * ouvs[ids, 1, None] \
            + corners[None, :, 1:] * ouvs[ids, 2, None]
        errors = (predicted - reference[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
        for i, index in enumerate(ids):
            comparator_rows.append({'cohort': 'lupori_guard', **records[index],
                'first_weak_five_site_gap_mm': float(errors[i, first[i]])})
    comparator_metrics = {cohort: {'n': len(panels[cohort]),
        'mean_first_rigid_gap_mm': float(np.mean([r['first_rigid_gap_mm'] for r in comparator_rows
            if r['cohort'] == cohort])),
        'first_rigid_over_5mm': sum(r['first_rigid_gap_mm'] > 5
            for r in comparator_rows if r['cohort'] == cohort),
        'top16_rigid_capture_1p5mm': sum(r['top16_rigid_oracle_mm'] <= 1.5
            for r in comparator_rows if r['cohort'] == cohort)} for cohort in panels}
    comparator_metrics['lupori_guard'] = {'n': len(guard_ids),
        'mean_first_weak_five_site_gap_mm': float(np.mean([
            r['first_weak_five_site_gap_mm'] for r in comparator_rows
            if r['cohort'] == 'lupori_guard']))}
(RUN / 'frozen_309_comparator.json').write_text(json.dumps({
    'checkpoint_sha256': comparator_sha, 'metrics': comparator_metrics,
    'rows': comparator_rows}, allow_nan=False) + '\n', encoding='utf8')
del old
torch.cuda.empty_cache()

torch.manual_seed(SEED + 3)
torch.cuda.manual_seed_all(SEED + 3)
model = C4JointSliceModel(side=SIDE).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
train_rng = np.random.default_rng(SEED + 4)
site_rng = torch.Generator(device='cuda').manual_seed(SEED + 5)
started, best_score, best_step = time.perf_counter(), float('inf'), None
readouts, stage_gates, stop_reason = {}, {}, None

with (RUN / 'train_draws.jsonl').open('w', encoding='utf8') as draw_file, \
        (RUN / 'training.jsonl').open('w', encoding='utf8') as training_file, \
        (RUN / 'readouts.jsonl').open('w', encoding='utf8') as readout_file:
    for step in range(UPDATES + 1):
        if step:
            samples = []
            with torch.no_grad():
                for slot in range(4):
                    while True:
                        base = int(train_rng.integers(56))
                        variant = int(train_rng.integers(VARIANTS))
                        seed = int(train_rng.integers(2**63 - 1, dtype=np.int64))
                        sample = sample_fov_balance_v10(context,
                            [base * VARIANTS + variant], seed, SIDE)
                        accepted = bool(sample['eligible'][0])
                        provenance = sample['provenance'][0]
                        assert provenance['split'] == 'train' and provenance['virtual_index'] == base * VARIANTS + variant
                        draw_file.write(json.dumps({'step': step, 'slot': slot,
                            'base': base, 'variant': variant, 'seed': seed,
                            'accepted': accepted, 'provenance': provenance}, allow_nan=False) + '\n')
                        if accepted:
                            samples.append(sample)
                            break
            image = torch.cat([sample['inputs'] for sample in samples])
            target_state = torch.cat([sample['state'] for sample in samples])
            target_reflection = torch.cat([sample['reflection'] for sample in samples])
            valid = torch.cat([sample['valid_mask'] for sample in samples])
            centre = torch.cat([sample['centre'] for sample in samples])
            pixel = torch.multinomial(valid.flatten(1).float(), 64,
                                      replacement=True, generator=site_rng)
            chart = torch.stack((pixel.remainder(SIDE),
                pixel.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            truth = centre.flatten(1, 2).gather(1, pixel[..., None].expand(-1, -1, 3))
            prediction = model.predict(image)
            scores = branch_scores(prediction)
            predicted = rigid_candidates(model, prediction, chart)
            site_um = (predicted - truth[:, None, None]).norm(dim=-1).mean(-1)
            five = rigid_candidates(model, prediction, corners[None].expand(4, -1, -1))
            true_five = rigid_points_090(target_state, target_reflection,
                                         corners, (SIDE, SIDE))
            five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            target_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
            normal_um = 4000 * (1 - (normal * target_normal[:, None]).sum(-1).abs().clamp_max(1))
            cost = (.75 * site_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-cost.detach() / .7, -1)
            nearest_anchor = (target_normal @ model.normal_frames[:, :, 2].T).abs().topk(4, -1).indices
            near_cost = cost.reshape(4, 4, model.normal_modes, 2).min(-1).values.gather(
                2, nearest_anchor[:, None].expand(-1, 4, -1))
            pose_loss = (-1.5 * torch.logsumexp(scores - cost / 1.5, -1)
                + .5 * F.kl_div(scores, target_mass, reduction='none').sum(-1)
                + .5 * near_cost.mean((1, 2))).mean()

            near = cost.detach().argmin(-1)
            first = scores.argmax(-1)
            mode = torch.stack((near // 2, first // 2,
                torch.zeros_like(near)), -1)
            reflection = torch.stack((near % 2, first % 2, target_reflection), -1)
            rows = torch.arange(4, device='cuda')
            selected_state = torch.stack((prediction['state'][rows, near // 2],
                prediction['state'][rows, first // 2], target_state), 1)
            mapped = model.map_sites(prediction, chart, mode, reflection, state=selected_state)
            map_gap = (mapped['coordinates'] - truth[:, None]).square().sum(-1).add(250**2).sqrt().sub(250)
            map_gap = map_gap.mean(-1) / 1000
            map_loss = (map_gap[:, 2] + .5 * map_gap[:, 0] + .2 * map_gap[:, 1]).mean()
            support = valid.float()
            pos_weight = ((1 - support).sum() / support.sum()).clamp(1, 10)
            support_loss = F.binary_cross_entropy_with_logits(
                mapped['correspondence_logit'][:, 2, 0], support, pos_weight=pos_weight)
            warp_penalty = mapped['local_displacement_um'][:, 2].square().mean().sqrt() / 1000
            loss = pose_loss + map_loss + .1 * support_loss + .02 * warp_penalty
            pose_map_gradient = None
            if step == 1:
                map_gradient = torch.autograd.grad(map_gap[:, 0].mean(),
                    model.pose[-1].weight, retain_graph=True)[0]
                pose_map_gradient = float(map_gradient.norm())
                assert math.isfinite(pose_map_gradient) and pose_map_gradient > 0

            acquired_loss = None
            if step % 4 == 0:
                animal = TRAIN[(step // 4 - 1) % len(TRAIN)]
                index = int(train_rng.choice(train_ids[animal]))
                draw_file.write(json.dumps({'step': step, 'slot': 4, 'kind': 'lupori_weak',
                    **records[index]}, allow_nan=False) + '\n')
                acquired_image = torch.zeros(1, 5, SIDE, SIDE, device='cuda')
                acquired_image[0, 0] = images[index]
                acquired = model.predict(acquired_image)
                acquired_scores = branch_scores(acquired)
                acquired_five = rigid_candidates(model, acquired, corners[None])
                reference = ouvs[index, 0] + corners[:, :1] * ouvs[index, 1] \
                    + corners[:, 1:] * ouvs[index, 2]
                five_um = (acquired_five[0] - reference).norm(dim=-1).mean(-1)
                weak_normal = F.normalize(torch.cross(ouvs[index, 1], ouvs[index, 2], dim=-1), dim=-1)
                estimated_normal = full_frame_state_to_components(acquired['state'])[1][0, :, :, 2]
                normal_um = 4000 * (1 - (estimated_normal * weak_normal).sum(-1).abs().clamp_max(1))
                weak_cost = ((five_um - 1000).clamp_min(0) +
                    (normal_um[:, None] - 100).clamp_min(0)).flatten() / 1000
                weak_target = F.softmax(-weak_cost.detach() / .7, -1)
                nearest_anchor = (weak_normal @ model.normal_frames[:, :, 2].T).abs().topk(4).indices
                near_weak = weak_cost.reshape(4, model.normal_modes, 2).min(-1).values[:, nearest_anchor]
                acquired_loss = (-1.5 * torch.logsumexp(acquired_scores[0] - weak_cost / 1.5, 0)
                    + .5 * F.kl_div(acquired_scores[0], weak_target, reduction='sum')
                    + .5 * near_weak.mean())
                loss = loss + .3 * acquired_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            training_file.write(json.dumps({'batch': step, 'synthetic_sections': 4,
                'pose_loss': float(pose_loss.detach()), 'map_loss': float(map_loss.detach()),
                'near_predicted_mapped_gap_mm': float(map_gap[:, 0].mean().detach()),
                'first_predicted_mapped_gap_mm': float(map_gap[:, 1].mean().detach()),
                'true_state_mapped_gap_mm': float(map_gap[:, 2].mean().detach()),
                'support_bce': float(support_loss.detach()),
                'warp_rms_mm': float(warp_penalty.detach()),
                'acquired_weak_loss': None if acquired_loss is None else float(acquired_loss.detach()),
                'pose_head_gradient_from_near_map': pose_map_gradient,
                'gradient_norm': float(gradient), 'loss': float(loss.detach()),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')

        if step in STAGES:
            model.eval()
            metrics, rows = evaluate(model)
            readouts[step] = metrics
            score = (metrics['inner']['mean_first_mapped_gap_mm'] +
                .5 * metrics['steep']['mean_first_mapped_gap_mm'] +
                .5 * metrics['sparse']['mean_first_mapped_gap_mm'])
            readout_file.write(json.dumps({'batch': step, 'selection_score': score,
                'metrics': metrics, 'rows': rows}, allow_nan=False) + '\n')
            readout_file.flush()
            training_file.flush()
            draw_file.flush()
            checkpoint = {'batch': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'config': config,
                'selection_score': score, 'calibrated': False}
            torch.save(checkpoint, RUN / f'joint_{step:05d}.pt')
            if step == 1000:
                previous = readouts[0]
                stage_gates[str(step)] = {
                    'inner_mapped_gain_ge0p35mm': metrics['inner']['mean_first_mapped_gap_mm'] <=
                        previous['inner']['mean_first_mapped_gap_mm'] - .35,
                    'inner_tail_plus_le2': metrics['inner']['first_mapped_over_5mm'] <=
                        previous['inner']['first_mapped_over_5mm'] + 2,
                    'steep_regression_le0p30mm': metrics['steep']['mean_first_mapped_gap_mm'] <=
                        previous['steep']['mean_first_mapped_gap_mm'] + .30,
                    'sparse_regression_le0p30mm': metrics['sparse']['mean_first_mapped_gap_mm'] <=
                        previous['sparse']['mean_first_mapped_gap_mm'] + .30,
                    'al1a_regression_le0p50mm': metrics['lupori_guard']['mean_first_weak_five_site_gap_mm'] <=
                        previous['lupori_guard']['mean_first_weak_five_site_gap_mm'] + .50}
            elif step == 4000:
                previous = readouts[1000]
                stage_gates[str(step)] = {
                    'inner_mapped_gain_ge0p15mm': metrics['inner']['mean_first_mapped_gap_mm'] <=
                        previous['inner']['mean_first_mapped_gap_mm'] - .15,
                    'inner_tail_plus_le2': metrics['inner']['first_mapped_over_5mm'] <=
                        previous['inner']['first_mapped_over_5mm'] + 2,
                    'steep_regression_le0p30mm': metrics['steep']['mean_first_mapped_gap_mm'] <=
                        previous['steep']['mean_first_mapped_gap_mm'] + .30,
                    'sparse_regression_le0p30mm': metrics['sparse']['mean_first_mapped_gap_mm'] <=
                        previous['sparse']['mean_first_mapped_gap_mm'] + .30,
                    'al1a_regression_le0p30mm': metrics['lupori_guard']['mean_first_weak_five_site_gap_mm'] <=
                        previous['lupori_guard']['mean_first_weak_five_site_gap_mm'] + .30,
                    'warp_not_worse_than_rigid_0p10mm': metrics['inner']['mean_first_mapped_gap_mm'] <=
                        metrics['inner']['mean_first_rigid_gap_mm'] + .10}
            elif step == 12000:
                stage_gates[str(step)] = {
                    'inner_mapped_beats_309_by0p35mm': metrics['inner']['mean_first_mapped_gap_mm'] <=
                        comparator_metrics['inner']['mean_first_rigid_gap_mm'] - .35,
                    'inner_rigid_beats_309_by0p15mm': metrics['inner']['mean_first_rigid_gap_mm'] <=
                        comparator_metrics['inner']['mean_first_rigid_gap_mm'] - .15,
                    'inner_map_beats_own_rigid_by0p05mm': metrics['inner']['mean_first_mapped_gap_mm'] <=
                        metrics['inner']['mean_first_rigid_gap_mm'] - .05,
                    'inner_tail_not_worse_than_309': metrics['inner']['first_mapped_over_5mm'] <=
                        comparator_metrics['inner']['first_rigid_over_5mm'],
                    'steep_mapped_regression_le0p20mm': metrics['steep']['mean_first_mapped_gap_mm'] <=
                        comparator_metrics['steep']['mean_first_rigid_gap_mm'] + .20,
                    'sparse_mapped_regression_le0p20mm': metrics['sparse']['mean_first_mapped_gap_mm'] <=
                        comparator_metrics['sparse']['mean_first_rigid_gap_mm'] + .20,
                    'al1a_regression_le0p20mm': metrics['lupori_guard']['mean_first_weak_five_site_gap_mm'] <=
                        comparator_metrics['lupori_guard']['mean_first_weak_five_site_gap_mm'] + .20}
            passed = all(stage_gates[str(step)].values()) if step in stage_gates else True
            if passed and score < best_score:
                best_score, best_step = score, step
                torch.save(checkpoint, RUN / 'best.pt')
            print(json.dumps({'event': 'stage', 'batch': step, 'metrics': metrics,
                'selection_score': score, 'best_batch': best_step}), flush=True)
            model.train()
            if step in stage_gates:
                print(json.dumps({'event': 'stage_gate', 'batch': step,
                    'passed': passed, 'clauses': stage_gates[str(step)]}), flush=True)
                if not passed and step < UPDATES:
                    stop_reason = f'predeclared batch-{step} development gate failed'
                    break

(RUN / 'completed.json').write_text(json.dumps({'candidate': 321,
    'planned_optimizer_batches': UPDATES, 'terminal_batch': step,
    'stop_reason': stop_reason, 'stage_gates': stage_gates,
    'final_internal_gate_passed': bool(stage_gates.get(str(UPDATES))) and
        all(stage_gates[str(UPDATES)].values()),
    'best_batch': best_step, 'best_score': best_score,
    'output_sha256': {path.name: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                      for path in RUN.iterdir() if path.name != 'completed.json'},
    'calibrated': False, 'model_promoted': False}, indent=2) + '\n', encoding='utf8')
