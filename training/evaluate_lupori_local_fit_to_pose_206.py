"""Post-exit, fixed 206 readout. Do not execute while either training arm is active."""

import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.arbitrary_plane_geometry import quicknii_to_allen_points
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
code = source.parent
pre_protocol = source.parents[1] / 'docs/publication/LUPORI_LOCAL_FIT_TO_POSE_206_PROTOCOL_20261011.md'
readout_protocol = source.parents[1] / 'docs/publication/LUPORI_LOCAL_FIT_TO_POSE_206_READOUT_PROTOCOL_20261011.md'
split_path = code / 'lupori_animal_split_185.json'
archive = root / 'data/lupori_joint_185/rawData.zip'
train = root / 'runs/lupori_local_fit_to_pose_206_matched_001'
parent_run = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A'
parent = parent_run / 'joint_step_00800.pt'
out = root / 'runs/lupori_local_fit_to_pose_206_readout_001'
panels = {'v3': root / 'data/joint_in_path_correspondence_128_dev_panel',
          'v4': root / 'data/fresh_v4_pose_dev_panel_132'}
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
steps, arms, side = (0, 200, 400), ('detached', 'fit'), 256
assert source.drive.upper() == root.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# These are post-exit checks; no run artifact may be read before the process exits.
assert all((train / arm / 'completed.json').is_file() for arm in arms)
training_config = json.loads((train / 'config.json').read_text())
parent_done = json.loads((parent_run / 'completed.json').read_text())
split = json.loads(split_path.read_text())
assert tuple(training_config['checkpoints']) == steps
assert training_config['updates_per_arm'] == 400
assert tuple(training_config['lupori_train_animals']) == ('CC2B', 'CC4B', 'CC4A')
assert training_config['inner_heldout_animal'] == 'AL1A'
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert split['development'] == ['CC1B'] and split['calibration'] == ['CC3A']
assert split['final_test'] == ['CC1A']
assert archive.stat().st_size == split['archive_bytes']
assert sha(split_path) == training_config['split_sha256']
assert sha(pre_protocol) == training_config['protocol_sha256']
assert sha(parent) == training_config['parent_checkpoint_sha256'] == parent_done['checkpoint_sha256']['800']
assert sha(parent_run / 'completed.json') == training_config['parent_completion_sha256']
assert all(sha(code / name) == digest for name, digest in training_config['source_sha256'].items())
assert not any(training_config[key] for key in
    ('calibrated', 'public_benchmark_used', 'development_calibration_final_read'))
assert not any(parent_done.get(key, False) for key in
    ('calibrated', 'public_benchmark_used', 'external_pretrained_weights_used',
     'heldout_animal_used_in_training'))

done, checkpoints, logs, draws = {}, {}, {}, {}
for arm in arms:
    directory = train / arm
    receipt = json.loads((directory / 'completed.json').read_text())
    assert receipt['arm'] == arm and receipt['updates'] == 400
    assert receipt['synthetic_presentations'] == 1200 and receipt['lupori_presentations'] == 400
    assert tuple(receipt['lupori_gradient_animals']) == ('CC2B', 'CC4B', 'CC4A')
    assert receipt['parent_checkpoint_sha256'] == sha(parent)
    assert receipt['config_sha256'] == sha(train / 'config.json')
    assert receipt['draws_sha256'] == sha(directory / 'draws.jsonl')
    assert receipt['training_sha256'] == sha(directory / 'training.jsonl')
    assert set(receipt['checkpoint_sha256']) == {'0', '200', '400'}
    assert not any(receipt[key] for key in
        ('calibrated', 'public_benchmark_used', 'development_calibration_final_read'))
    for step in steps:
        path = directory / f'joint_step_{step:05d}.pt'
        assert sha(path) == receipt['checkpoint_sha256'][str(step)]
        checkpoints[arm, step] = path
    rows = [json.loads(line) for line in (directory / 'training.jsonl').open()]
    assert len(rows) == 400 and [row['step'] for row in rows] == list(range(1, 401))
    assert all(row['arm'] == arm and row['all_gradients_finite'] for row in rows)
    if arm == 'detached':
        assert all(row['fit_applied_geometry_output_gradient_norm'] == 0 for row in rows)
    else:
        assert all(row['fit_probability_row_gradient_max'] == 0 for row in rows if row['fit_eligible'])
        assert all(row['fit_applied_geometry_output_gradient_norm'] <=
                   .2 * row['supervised_geometry_output_gradient_norm'] + 1e-5
                   for row in rows if row['fit_eligible'])
    assert sum(row['fit_eligible'] for row in rows) == receipt['fit_eligible_presentations']
    logs[arm] = rows
    draws[arm] = [json.loads(line) for line in (directory / 'draws.jsonl').open()]
    done[arm] = receipt
assert done['detached']['draws_sha256'] == done['fit']['draws_sha256']
assert draws['detached'] == draws['fit']
draw_rows = draws['detached']
assert len([row for row in draw_rows if row['kind'] == 'synthetic_train']) == done['detached']['synthetic_draw_attempts'] == done['fit']['synthetic_draw_attempts']
assert len([row for row in draw_rows if row['kind'] == 'synthetic_train' and row['used']]) == 1200
assert len([row for row in draw_rows if row['kind'] == 'lupori_train']) == 400
assert all(row['animal_id'] in ('CC2B', 'CC4B', 'CC4A')
           for row in draw_rows if row['kind'] == 'lupori_train')
assert len({(row['animal_id'], row['section_id']) for row in draw_rows
            if row['kind'] == 'lupori_train'}) == training_config['lupori_sections'] == 188
train_synthetic_ids = {row['physical_section_id'] for row in draw_rows
                       if row['kind'] == 'synthetic_train'}

panel_records, panel_done = {}, {}
for cohort, panel in panels.items():
    receipt = json.loads((panel / 'completed.json').read_text())
    frozen = json.loads((panel / 'protocol.json').read_text())
    rows = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == receipt['protocol_sha256']
    assert sha(panel / 'records.jsonl') == receipt['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in frozen['source_sha256'].items())
    assert len(rows) == receipt['physical_sections'] == 256
    assert sum(row['eligible'] for row in rows) == receipt['eligible']
    assert len({row['synthetic_subject_plan_id'] for row in rows}) == 8
    assert len({row['panel_physical_section_id'] for row in rows}) == 256
    assert all(row['provenance']['split'] == 'development' for row in rows)
    assert all(sha(panel / row['file']) == row['sha256'] for row in rows)
    panel_records[cohort], panel_done[cohort] = rows, receipt
assert panel_done['v3']['plan_completed_sha256'] == panel_done['v4']['plan_completed_sha256']
assert not {row['panel_physical_section_id'] for row in panel_records['v3']} & {
    row['panel_physical_section_id'] for row in panel_records['v4']}

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert all(sha(coronal / name) == coronal_done['output_sha256'][name]
           for name in ('images.npy', 'geometry.npz', 'records.jsonl'))
coronal_all = [json.loads(line) for line in (coronal / 'records.jsonl').open()]
coronal_records = [row for row in coronal_all if row['training_split'] == 'development']
assert len(coronal_records) == 64 and len({row['animal_id'] for row in coronal_records}) == 6
assert not {row['animal_id'] for row in coronal_records} & {
    row['animal_id'] for row in coronal_all if row['training_split'] == 'train'}
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()

sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert all(sha(sagittal / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_train_records = [json.loads(line) for line in
                          (sagittal_train / 'geometry.jsonl').open()]
assert len(sagittal_records) == 158 and len({row['donor_id'] for row in sagittal_records}) == 8
assert all(row['split'] == 'weak_dev2_source_available' for row in sagittal_records)
assert not {row['donor_id'] for row in sagittal_records} & {
    row['donor_id'] for row in sagittal_train_records}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')

five_pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                            [127.5, 127.5]], device='cuda')
five_chart = five_pixels / side
axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
flags = torch.arange(2, device='cuda')[None, None]

al1a_records, al1a_images, al1a_targets = [], [], []
with ZipFile(archive) as source_zip:
    info = ElementTree.fromstring(source_zip.read('AL1A/AL1A-info.xml'))
    valid = {row.findtext('name') for row in info.findall('slices')
             if row.findtext('valid') == '1'}
    visualign = json.loads(source_zip.read('AL1A/AL1A-visualign.json'))
    assert visualign['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
    for row in visualign['slices']:
        section_id = Path(row['filename']).name.removesuffix('-thumb.png')
        if section_id not in valid:
            continue
        image_bytes = source_zip.read(f'AL1A/{row["filename"]}')
        rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
            (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        q = torch.tensor(row['anchoring'], dtype=torch.float32).reshape(3, 3)
        points = q[0] + (five_pixels.cpu() / side)[:, :1] * q[1] + (
            five_pixels.cpu() / side)[:, 1:] * q[2]
        al1a_images.append(gray)
        al1a_targets.append(quicknii_to_allen_points(points).numpy() * 25)
        al1a_records.append({'animal_id': 'AL1A', 'section_id': section_id,
            'source_thumbnail': f'AL1A/{row["filename"]}',
            'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest()})
assert len(al1a_records) == len({row['section_id'] for row in al1a_records}) == 52
al1a_images = np.stack(al1a_images)
al1a_targets = torch.from_numpy(np.stack(al1a_targets)).cuda()

# The fresh panel has one eligible, never-trained physical draw per TRAIN base.
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == training_config['synthetic_provenance']
v7_panel, v7_attempts = [], []
for base in range(64):
    virtual = base * 64
    for attempt in range(16):
        draw_seed = 206700000000 + 16 * base + attempt
        with torch.no_grad():
            sample = sample_one_shot_slide_artifacts_v7(context, [virtual], draw_seed, side=side)
        physical_id = sample['provenance'][0]['physical_section_id']
        assert physical_id not in train_synthetic_ids
        v7_attempts.append({'base_index': base, 'virtual_index': virtual,
            'draw_seed': draw_seed, 'attempt': attempt,
            'physical_section_id': physical_id, 'eligible': bool(sample['eligible'][0])})
        if bool(sample['eligible'][0]):
            break
    assert bool(sample['eligible'][0])
    v7_panel.append({'base_index': base, 'virtual_index': virtual,
        'draw_seed': draw_seed, 'attempt': attempt, 'physical_section_id': physical_id,
        'virtual_subject_id': sample['provenance'][0]['virtual_subject_id'],
        'base_subject_id': sample['provenance'][0]['base_lineage']['subject_id'],
        'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest(),
        'state_sha256': hashlib.sha256(sample['state'].cpu().numpy().tobytes()).hexdigest(),
        'valid_mask_sha256': hashlib.sha256(sample['valid_mask'].cpu().numpy().tobytes()).hexdigest(),
        'valid_pixels': int(sample['valid_mask'][0].sum()),
        'inputs': sample['inputs'].cpu(), 'state': sample['state'].cpu(),
        'reflection': sample['reflection'].cpu(), 'valid_mask': sample['valid_mask'].cpu()})
assert len(v7_panel) == len({row['physical_section_id'] for row in v7_panel}) == 64

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
al1a_rows, synthetic_rows, old_real_rows = [], [], []
with torch.inference_mode():
    for step in steps:
        for arm in arms:
            # The locally produced, SHA-verified checkpoint includes NumPy RNG arrays.
            saved = torch.load(checkpoints[arm, step], map_location='cpu', weights_only=False)
            assert saved['step'] == step and saved['arm'] == arm
            assert saved['config'] == training_config and saved['calibrated'] is False
            model.load_state_dict(saved['model'], strict=True)
            del saved

            for start in range(0, 52, 8):
                batch = al1a_images[start:start + 8]
                inputs = torch.zeros(len(batch), 5, side, side, device='cuda')
                inputs[:, 0] = torch.from_numpy(batch).cuda()
                prediction = model.predict(inputs)
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                reflections = flags.expand(len(batch), model.modes, 2)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                mapped = rigid_points_090(states, reflections, five_chart)
                error = (mapped - al1a_targets[start:start + len(batch), None, None]
                         ).norm(dim=-1).mean(-1).flatten(1) / 1000
                chosen = prior.argmax(-1)
                top16 = prior.topk(16, -1).indices
                for j in range(len(batch)):
                    al1a_rows.append({**al1a_records[start + j], 'arm': arm, 'step': step,
                        'checkpoint_sha256': done[arm]['checkpoint_sha256'][str(step)],
                        'top1_branch_id': int(chosen[j]),
                        'direct_top1_five_point_mm': float(error[j, chosen[j]]),
                        'oracle_top16_five_point_mm': float(error[j, top16[j]].min()),
                        'oracle_all160_five_point_mm': float(error[j].min()),
                        'label_role': 'reserved TRAIN animal human VisuAlign weak reference'})

            for cohort, panel in panels.items():
                for record in (row for row in panel_records[cohort] if row['eligible']):
                    with np.load(panel / record['file'], allow_pickle=False) as arrays:
                        image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                        valid_mask = torch.from_numpy(arrays['valid_mask'].flatten().copy()).cuda().bool()
                        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                        reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
                    assert int(valid_mask.sum()) == record['valid_pixels']
                    prediction = model.predict(image)
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                    top_id = int(prior[0].argmax())
                    proposed = rigid_points_090(prediction['state'][:, top_id // 2],
                        torch.tensor([top_id % 2], device='cuda'), chart)[0]
                    target = rigid_points_090(truth, reflection, chart)[0]
                    synthetic_rows.append({'arm': arm, 'step': step, 'cohort': cohort,
                        'section_id': record['section_id'],
                        'physical_section_id': record['panel_physical_section_id'],
                        'subject_plan_id': record['synthetic_subject_plan_id'],
                        'panel_file_sha256': record['sha256'],
                        'valid_pixels': int(valid_mask.sum()), 'top1_branch_id': top_id,
                        'direct_top1_tissue_mm': float((proposed - target).norm(dim=-1)[valid_mask].mean() / 1000)})

            for item in v7_panel:
                image = item['inputs'].cuda()
                truth = item['state'].cuda()
                reflection = item['reflection'].cuda().long()
                valid_mask = item['valid_mask'][0].flatten().cuda().bool()
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                top_id = int(prior[0].argmax())
                proposed = rigid_points_090(prediction['state'][:, top_id // 2],
                    torch.tensor([top_id % 2], device='cuda'), chart)[0]
                target = rigid_points_090(truth, reflection, chart)[0]
                synthetic_rows.append({'arm': arm, 'step': step, 'cohort': 'v7_train_fresh',
                    'section_id': item['physical_section_id'],
                    'physical_section_id': item['physical_section_id'],
                    'base_subject_id': item['base_subject_id'],
                    'virtual_subject_id': item['virtual_subject_id'],
                    'draw_seed': item['draw_seed'], 'input_sha256': item['input_sha256'],
                    'valid_pixels': item['valid_pixels'], 'top1_branch_id': top_id,
                    'direct_top1_tissue_mm': float((proposed - target).norm(dim=-1)[valid_mask].mean() / 1000)})

            for family, records in (('coronal', coronal_records), ('sagittal', sagittal_records)):
                for record in records:
                    index = record['array_row_index']
                    if family == 'coronal':
                        native = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                            np.zeros((4, 192, 192), dtype=np.float32)))[None]
                        image = F.interpolate(torch.from_numpy(native).cuda(), (side, side),
                            mode='bilinear', align_corners=False)
                        affine = torch.as_tensor(coronal_affines[index], device='cuda', dtype=torch.float32)
                        old_pixels = (five_pixels + .5) * (192 / side) - .5
                        reference = affine[:, 2] + (old_pixels[:, :1] * affine[:, 0]
                            + old_pixels[:, 1:] * affine[:, 1])
                        donor = record['animal_id']
                    else:
                        native = np.concatenate((np.asarray(sagittal_images[index], dtype=np.float32),
                            np.zeros((4, side, side), dtype=np.float32)))[None]
                        image = torch.from_numpy(native).cuda()
                        affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                            device='cuda', dtype=torch.float32)
                        reference = affine[:, 2] + (five_pixels[:, :1] * affine[:, 0]
                            + five_pixels[:, 1:] * affine[:, 1])
                        donor = record['donor_id']
                    prediction = model.predict(image)
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                    top_id = int(prior[0].argmax())
                    proposed = rigid_points_090(prediction['state'][:, top_id // 2],
                        torch.tensor([top_id % 2], device='cuda'), five_chart)[0]
                    old_real_rows.append({'arm': arm, 'step': step, 'family': family,
                        'donor_id': donor, 'section_id': record['section_id'],
                        'specimen_id': record['specimen_id'],
                        'experiment_id': record['experiment_id'],
                        'top1_branch_id': top_id,
                        'direct_top1_weak_five_point_mm': float(
                            (proposed - reference).norm(dim=-1).mean() / 1000),
                        'label_role': 'inherited weak Allen affine, not expert truth'})
            print(json.dumps({'event': 'checkpoint_read', 'arm': arm, 'step': step}), flush=True)

assert len(al1a_rows) == 6 * 52
assert len(synthetic_rows) == 6 * (panel_done['v3']['eligible'] + panel_done['v4']['eligible'] + 64)
assert len(old_real_rows) == 6 * (64 + 158)
summary = {'scope': '206 post-exit descriptive readout; AL1A reserved TRAIN, frozen synthetic DEV v3/v4, fresh TRAIN-base v7, old weak-real donor-disjoint panels',
    'top1': 'argmax of joint mode-and-reflection prior before atlas matching, fitting, or reranking',
    'al1a_label': 'approximate human VisuAlign plane, not expert or histological truth',
    'v7_scope': 'one new eligible draw per each of 64 previously trained deformation bases; out-of-draw, not out-of-base or biological DEV',
    'training_fit_gradient_audit': {}, 'al1a': {}, 'synthetic': {},
    'old_weak_real': {}, 'advancement_gate_step400': {}}
for arm in arms:
    eligible = [row for row in logs[arm] if row['fit_eligible']]
    assert all(np.isfinite(row['fit_vs_weak_reference_state_gradient_dot']) and
               np.isfinite(row['fit_vs_weak_reference_state_gradient_cosine'])
               for row in eligible)
    summary['training_fit_gradient_audit'][arm] = {
        'eligible_presentations': len(eligible),
        'direction_positive_count': sum(row['fit_descent_predicted_to_reduce_weak_reference_cost']
                                        for row in eligible),
        'direction_positive_fraction': float(np.mean([
            row['fit_descent_predicted_to_reduce_weak_reference_cost'] for row in eligible])) if eligible else None,
        'direction_cosine_median': float(np.median([
            row['fit_vs_weak_reference_state_gradient_cosine'] for row in eligible])) if eligible else None,
        'applied_gradient_nonzero_count': sum(row['fit_applied_geometry_output_gradient_norm'] > 0
                                              for row in eligible)}
rng = np.random.default_rng(20671101)
for step in steps:
    paired = {}
    for arm in arms:
        rows = [row for row in al1a_rows if row['step'] == step and row['arm'] == arm]
        paired[arm] = {row['section_id']: row for row in rows}
        selected = np.array([row['direct_top1_five_point_mm'] for row in rows])
        oracle = np.array([row['oracle_all160_five_point_mm'] for row in rows])
        summary['al1a'].setdefault(str(step), {})[arm] = {
            'sections': len(rows), 'top1_mean_mm': float(selected.mean()),
            'top1_median_mm': float(np.median(selected)),
            'top1_gt5_count': int((selected > 5).sum()),
            'oracle_all160_mean_mm': float(oracle.mean())}
    assert paired['detached'].keys() == paired['fit'].keys()
    keys = sorted(paired['detached'])
    detached = np.array([paired['detached'][key]['direct_top1_five_point_mm'] for key in keys])
    fitted = np.array([paired['fit'][key]['direct_top1_five_point_mm'] for key in keys])
    oracle_detached = np.array([paired['detached'][key]['oracle_all160_five_point_mm'] for key in keys])
    oracle_fit = np.array([paired['fit'][key]['oracle_all160_five_point_mm'] for key in keys])
    indices = rng.integers(0, len(keys), size=(10000, len(keys)))
    delta = fitted - detached
    oracle_delta = oracle_fit - oracle_detached
    summary['al1a'][str(step)]['paired_fit_minus_detached'] = {
        'top1_mean_delta_mm': float(delta.mean()),
        'top1_section_bootstrap_95pct_mm': np.percentile(delta[indices].mean(1), [2.5, 97.5]).tolist(),
        'top1_gt5_count_delta': int((fitted > 5).sum() - (detached > 5).sum()),
        'top1_new_gt5_count': int(((fitted > 5) & (detached <= 5)).sum()),
        'top1_resolved_gt5_count': int(((fitted <= 5) & (detached > 5)).sum()),
        'oracle_all160_mean_delta_mm': float(oracle_delta.mean()),
        'oracle_all160_section_bootstrap_95pct_mm': np.percentile(
            oracle_delta[indices].mean(1), [2.5, 97.5]).tolist()}

for cohort in ('v3', 'v4', 'v7_train_fresh'):
    summary['synthetic'][cohort] = {}
    for step in steps:
        paired = {}
        for arm in arms:
            rows = [row for row in synthetic_rows
                    if row['cohort'] == cohort and row['step'] == step and row['arm'] == arm]
            paired[arm] = {row['physical_section_id']: row for row in rows}
            values = np.array([row['direct_top1_tissue_mm'] for row in rows])
            summary['synthetic'][cohort].setdefault(str(step), {})[arm] = {
                'sections': len(rows), 'section_weighted_mean_mm': float(values.mean()),
                'section_weighted_median_mm': float(np.median(values))}
            if cohort in panels:
                plans = sorted({row['subject_plan_id'] for row in rows})
                summary['synthetic'][cohort][str(step)][arm]['plan_equal_mean_mm'] = float(np.mean([
                    np.mean([row['direct_top1_tissue_mm'] for row in rows
                             if row['subject_plan_id'] == plan]) for plan in plans]))
        assert paired['detached'].keys() == paired['fit'].keys()
        keys = sorted(paired['detached'])
        detached = np.array([paired['detached'][key]['direct_top1_tissue_mm'] for key in keys])
        fitted = np.array([paired['fit'][key]['direct_top1_tissue_mm'] for key in keys])
        summary['synthetic'][cohort][str(step)]['paired_fit_minus_detached'] = {
            'section_mean_delta_mm': float((fitted - detached).mean()),
            'section_mean_ratio': float(fitted.mean() / detached.mean())}

for family in ('coronal', 'sagittal'):
    summary['old_weak_real'][family] = {}
    for step in steps:
        paired = {}
        for arm in arms:
            rows = [row for row in old_real_rows
                    if row['family'] == family and row['step'] == step and row['arm'] == arm]
            paired[arm] = {(row['donor_id'], row['section_id']): row for row in rows}
            donors = sorted({row['donor_id'] for row in rows})
            means = {donor: float(np.mean([row['direct_top1_weak_five_point_mm']
                for row in rows if row['donor_id'] == donor])) for donor in donors}
            summary['old_weak_real'][family].setdefault(str(step), {})[arm] = {
                'sections': len(rows), 'donors': len(donors),
                'section_weighted_mean_mm': float(np.mean([
                    row['direct_top1_weak_five_point_mm'] for row in rows])),
                'donor_equal_mean_mm': float(np.mean(list(means.values()))),
                'by_donor_mean_mm': means}
        assert paired['detached'].keys() == paired['fit'].keys()
        delta = (summary['old_weak_real'][family][str(step)]['fit']['donor_equal_mean_mm']
                 - summary['old_weak_real'][family][str(step)]['detached']['donor_equal_mean_mm'])
        summary['old_weak_real'][family][str(step)]['paired_fit_minus_detached'] = {
            'donor_equal_mean_delta_mm': float(delta)}

assert abs(summary['al1a']['0']['paired_fit_minus_detached']['top1_mean_delta_mm']) < 1e-5
assert all(abs(summary['synthetic'][cohort]['0']['paired_fit_minus_detached']['section_mean_delta_mm']) < 1e-5
           for cohort in ('v3', 'v4', 'v7_train_fresh'))
assert all(abs(summary['old_weak_real'][family]['0']['paired_fit_minus_detached']['donor_equal_mean_delta_mm']) < 1e-5
           for family in ('coronal', 'sagittal'))
gate = summary['advancement_gate_step400']
gate['al1a_top1_mean_improvement_ge_0p10mm'] = (
    summary['al1a']['400']['paired_fit_minus_detached']['top1_mean_delta_mm'] <= -.10)
gate['al1a_no_new_top1_gt5mm'] = (
    summary['al1a']['400']['paired_fit_minus_detached']['top1_new_gt5_count'] == 0)
for cohort in ('v3', 'v7_train_fresh'):
    gate[f'{cohort}_section_mean_ratio_le_1p10'] = (
        summary['synthetic'][cohort]['400']['paired_fit_minus_detached']['section_mean_ratio'] <= 1.10)
for family in ('coronal', 'sagittal'):
    gate[f'{family}_donor_equal_delta_le_0p20mm'] = (
        summary['old_weak_real'][family]['400']['paired_fit_minus_detached']['donor_equal_mean_delta_mm'] <= .20)
gate['passes_all_predeclared_checks'] = all(gate.values())
summary['calibrated'] = summary['official_lupori_dev_cal_final_read'] = summary['public_benchmark_used'] = False
summary['bootstrap_interpretation'] = 'descriptive resampling of these 52 paired AL1A sections only; no animal-population interval'
summary['model_selection'] = 'none; all 0/200/400 milestones reported, no checkpoint chosen'

receipt_config = {'evaluator_source_sha256': sha(source),
    'readout_protocol_sha256': sha(readout_protocol),
    'training_protocol_sha256': sha(pre_protocol),
    'training_config_sha256': sha(train / 'config.json'),
    'training_completion_sha256': {arm: sha(train / arm / 'completed.json') for arm in arms},
    'training_checkpoint_sha256': {arm: done[arm]['checkpoint_sha256'] for arm in arms},
    'training_draws_sha256': done['detached']['draws_sha256'],
    'parent_checkpoint_sha256': sha(parent), 'split_sha256': sha(split_path),
    'lupori_archive_bytes': split['archive_bytes'], 'lupori_archive_md5': split['archive_md5'],
    'panel_completion_sha256': {cohort: sha(panel / 'completed.json') for cohort, panel in panels.items()},
    'panel_records_sha256': {cohort: sha(panel / 'records.jsonl') for cohort, panel in panels.items()},
    'coronal_completion_sha256': sha(coronal / 'completed.json'),
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'v7_first_seed': 206700000000, 'v7_max_attempts_per_base': 16,
    'v7_virtual_index': 'base_index * 64, base_index = 0..63',
    'bootstrap_seed': 20671101, 'bootstrap_draws': 10000,
    'calibrated': False, 'official_lupori_dev_cal_final_read': False,
    'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(receipt_config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for name, rows in (('al1a_rows.jsonl', al1a_rows),
                   ('synthetic_rows.jsonl', synthetic_rows),
                   ('old_weak_real_rows.jsonl', old_real_rows),
                   ('v7_fresh_panel.jsonl', [{key: value for key, value in row.items()
                       if key not in ('inputs', 'state', 'reflection', 'valid_mask')}
                       for row in v7_panel]),
                   ('v7_fresh_draws.jsonl', v7_attempts)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({
    'training_completion_sha256': receipt_config['training_completion_sha256'],
    'checkpoint_sha256': receipt_config['training_checkpoint_sha256'],
    'evaluator_source_sha256': receipt_config['evaluator_source_sha256'],
    'readout_protocol_sha256': receipt_config['readout_protocol_sha256'],
    'al1a_sections': 52, 'v7_fresh_train_base_sections': 64,
    'synthetic_v3_eligible': panel_done['v3']['eligible'],
    'synthetic_v4_eligible': panel_done['v4']['eligible'],
    'coronal_sections': 64, 'sagittal_sections': 158,
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'summary.json', 'al1a_rows.jsonl', 'synthetic_rows.jsonl',
         'old_weak_real_rows.jsonl', 'v7_fresh_panel.jsonl', 'v7_fresh_draws.jsonl')},
    'calibrated': False, 'official_lupori_dev_cal_final_read': False,
    'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'readout_complete', 'advancement_gate_step400': gate}), flush=True)
