"""Frozen TRAIN-inner 226-versus-233 source-contrast readout; no optimization."""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import quicknii_to_allen_points
from training.arbitrary_plane_one_shot_inference import infer_one_shot
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.kimlab_lsfm_appearance_232 import bind_lsfm_appearance_232, select_atlas_appearance_232
from training.support_aware_framing_217 import sample_support_aware_framing_217


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def without_appearance_source(value):
    if isinstance(value, dict):
        return {key: without_appearance_source(item) for key, item in value.items()
                if key not in ('atlas_binding', 'atlas_appearance_source', 'appearance_source')}
    if isinstance(value, list):
        return [without_appearance_source(item) for item in value]
    return value


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/KIMLAB_LSFM_SOURCE_ABLATION_233_PROTOCOL_20261011.md'
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
control = root / 'runs/pose_appearance_exposure_226_train_001'
intervention = root / 'runs/kimlab_lsfm_source_233_train_001'
readout = root / 'runs/pose_appearance_exposure_226_train_readout_001'
parent = root / 'runs/pose_framing_221_framed_train_001'
out = root / 'runs/kimlab_lsfm_source_233_readout_001'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert archive.stat().st_size == split['archive_bytes']
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

done = {arm: json.loads((directory / 'completed.json').read_text())
        for arm, directory in (('226', control), ('233', intervention))}
config = {arm: json.loads((directory / 'config.json').read_text())
          for arm, directory in (('226', control), ('233', intervention))}
parent_done = json.loads((parent / 'completed.json').read_text())
parent_hash = sha(parent / 'joint_step_00400.pt')
assert parent_hash == parent_done['checkpoint_sha256']['400']
assert all(done[arm]['updates'] == 1200 and done[arm]['parent_checkpoint_sha256'] == parent_hash
           and sha(directory / 'config.json') == done[arm]['config_sha256']
           and not any(done[arm][key] for key in ('calibrated', 'public_benchmark_used',
               'external_pretrained_weights_used', 'heldout_animal_used_in_training'))
           for arm, directory in (('226', control), ('233', intervention)))
assert done['233']['protocol_sha256'] == sha(protocol)
assert config['226']['seed'] == config['233']['seed'] == 20261011226
assert config['226']['lupori_gradient_animals'] == config['233']['lupori_gradient_animals']
assert config['233']['inner_guard_animal'] == 'AL1A'
assert config['233']['kimlab_lsfm_source_ablation_233'] is True
assert config['233']['matched_control_completion_sha256'] == sha(control / 'completed.json')
assert all(sha(source.parent / name) == digest
           for name, digest in done['233']['source_sha256'].items())
checkpoints = [(f'{arm}_{step:04d}', directory / f'joint_step_{step:05d}.pt')
               for arm, directory in (('226', control), ('233', intervention))
               for step in (0, 400, 800, 1200)]
checkpoint_hashes = {label: sha(path) for label, path in checkpoints}
assert all(checkpoint_hashes[f'{arm}_{step:04d}'] == done[arm]['checkpoint_sha256'][str(step)]
           for arm in ('226', '233') for step in (0, 400, 800, 1200))
zero_control = torch.load(control / 'joint_step_00000.pt', map_location='cpu', weights_only=True)
zero_intervention = torch.load(intervention / 'joint_step_00000.pt', map_location='cpu', weights_only=True)
assert zero_control['step'] == zero_intervention['step'] == 0
assert all(torch.equal(value, zero_intervention['model'][key])
           for key, value in zero_control['model'].items())
del zero_control, zero_intervention

draw_rows = {}
for arm, directory in (('226', control), ('233', intervention)):
    assert sha(directory / 'draws.jsonl') == done[arm]['draws_sha256']
    draw_rows[arm] = [json.loads(line) for line in (directory / 'draws.jsonl').open()]
matched_draws = [without_appearance_source(row) for row in draw_rows['226']] == [
    without_appearance_source(row) for row in draw_rows['233']]
first_draw_mismatch = next((index for index, (a, b) in enumerate(zip(
    draw_rows['226'], draw_rows['233']))
    if without_appearance_source(a) != without_appearance_source(b)), None)
if first_draw_mismatch is None and len(draw_rows['226']) != len(draw_rows['233']):
    first_draw_mismatch = min(len(draw_rows['226']), len(draw_rows['233']))
source_counts = {label: sum(row.get('atlas_appearance_source') == label
                            for row in draw_rows['233'] if row['kind'] == 'synthetic_train'
                            and row['slot'] < 2 and row['used'])
                 for label in ('Allen average', 'KimLab LSFM')}
assert all(source_counts.values())

readout_done = json.loads((readout / 'completed.json').read_text())
cases_path = readout / 'cases.jsonl'
assert sha(cases_path) == readout_done['output_sha256']['cases.jsonl']
frozen_rows = [json.loads(line) for line in cases_path.open()]
frozen_rows = [row for row in frozen_rows if row['panel'] == 'v7_framed']
assert len(frozen_rows) == 64 and len({row['physical_section_id'] for row in frozen_rows}) == 64
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == config['226']['synthetic_provenance']
context = bind_lsfm_appearance_232(context)
assert context['lsfm_binding_232'] == config['233']['kimlab_lsfm_binding_232']

cases, attempts = [], []
with torch.inference_mode():
    for row in frozen_rows:
        base, seed = int(row['base']), int(row['seed'])
        sampled = sample_panel_geometry_train_166(context, [base], seed, side=256)
        sampled = sample_support_aware_framing_217(sampled, seed)
        sampled = sample_one_shot_slide_artifacts_v7(
            context, [base], seed, side=256, source_sample=sampled)
        image_hash = hashlib.sha256(sampled['inputs'].cpu().numpy().tobytes()).hexdigest()
        assert bool(sampled['eligible'][0])
        assert sampled['provenance'][0]['physical_section_id'] == row['physical_section_id']
        assert image_hash == row['input_sha256']
        pixels = sampled['valid_mask'][0].flatten().nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
            device='cuda').long()]
        sites = torch.stack((pixels.remainder(256), pixels.div(256, rounding_mode='floor')), -1).float() / 256
        cases.append({'panel': '226_frozen_v7', 'base': base, 'seed': seed,
            'physical_section_id': row['physical_section_id'], 'input_sha256': image_hash,
            'image': sampled['inputs'].cpu(), 'offsets': sampled['offsets'].cpu(),
            'weights': sampled['weights'].cpu(), 'sites': sites.cpu(), 'pixels': pixels.cpu(),
            'truth': sampled['centre'][0].reshape(-1, 3)[pixels].cpu(),
            'normal': full_frame_state_to_components(sampled['state'])[1][0, :, 2].cpu()})
    for base in (60, 61, 62, 63):
        accepted = attempt = 0
        while accepted < 16:
            seed = 202610112330000 + 10000 * base + 1009 * attempt
            draw_context, source_label = select_atlas_appearance_232(context, seed)
            record = {'base': base, 'attempt': attempt, 'seed': seed,
                      'source': source_label, 'used': False}
            if source_label == 'KimLab LSFM':
                sampled = sample_panel_geometry_train_166(draw_context, [base], seed, side=256)
                sampled = sample_support_aware_framing_217(sampled, seed)
                sampled = sample_one_shot_slide_artifacts_v7(
                    draw_context, [base], seed, side=256, source_sample=sampled)
                record['used'] = bool(sampled['eligible'][0])
                record['physical_section_id'] = sampled['provenance'][0]['physical_section_id']
                if record['used']:
                    assert sampled['provenance'][0]['atlas_binding'] == context['lsfm_binding_232']
                    pixels = sampled['valid_mask'][0].flatten().nonzero()[:, 0]
                    pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
                        device='cuda').long()]
                    sites = torch.stack((pixels.remainder(256),
                        pixels.div(256, rounding_mode='floor')), -1).float() / 256
                    image_hash = hashlib.sha256(sampled['inputs'].cpu().numpy().tobytes()).hexdigest()
                    record['input_sha256'] = image_hash
                    cases.append({'panel': '233_new_lsfm_v7', 'base': base, 'seed': seed,
                        'physical_section_id': record['physical_section_id'],
                        'input_sha256': image_hash, 'image': sampled['inputs'].cpu(),
                        'offsets': sampled['offsets'].cpu(), 'weights': sampled['weights'].cpu(),
                        'sites': sites.cpu(), 'pixels': pixels.cpu(),
                        'truth': sampled['centre'][0].reshape(-1, 3)[pixels].cpu(),
                        'normal': full_frame_state_to_components(sampled['state'])[1][0, :, 2].cpu()})
                    accepted += 1
            attempts.append(record)
            attempt += 1
assert len(cases) == 128 and len({row['physical_section_id'] for row in cases}) == 128

five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / 256
al1a_cases = []
with ZipFile(archive) as zipped:
    info_bytes = zipped.read('AL1A/AL1A-info.xml')
    visualign_bytes = zipped.read('AL1A/AL1A-visualign.json')
    valid = {row.findtext('name') for row in ElementTree.fromstring(info_bytes).findall('slices')
             if row.findtext('valid') == '1'}
    visualign = json.loads(visualign_bytes)
    assert visualign['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
    for row in visualign['slices']:
        section_id = Path(row['filename']).name.removesuffix('-thumb.png')
        if section_id not in valid:
            continue
        thumbnail = f'AL1A/{row["filename"]}'
        image_bytes = zipped.read(thumbnail)
        rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
            (256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        origin, u, v = torch.tensor(row['anchoring'], dtype=torch.float32).reshape(3, 3)
        quicknii = origin + five.cpu()[:, :1] * u + five.cpu()[:, 1:] * v
        image = torch.zeros(1, 5, 256, 256)
        image[0, 0] = torch.from_numpy(gray)
        al1a_cases.append({'animal_id': 'AL1A', 'specimen_id': 'AL1A',
            'experiment_id': None, 'source_section_id': section_id,
            'source_thumbnail': thumbnail,
            'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
            'image': image, 'truth': quicknii_to_allen_points(quicknii).cuda() * 25})
assert len(al1a_cases) == 52

out.mkdir(parents=True)
manifest = {'schema': 'anatomy-tracker.kimlab-source-ablation/233',
    'protocol_sha256': sha(protocol), 'script_sha256': sha(source),
    'split_sha256': sha(split_path), 'source_readout_cases_sha256': sha(cases_path),
    'source_al1a_info_sha256': hashlib.sha256(info_bytes).hexdigest(),
    'source_al1a_visualign_sha256': hashlib.sha256(visualign_bytes).hexdigest(),
    'source_lsfm_binding': context['lsfm_binding_232'],
    'training_completion_sha256': {arm: sha(directory / 'completed.json')
        for arm, directory in (('226', control), ('233', intervention))},
    'checkpoint_sha256': checkpoint_hashes, 'matched_draws': matched_draws,
    'first_draw_mismatch': first_draw_mismatch, 'draw_counts': {arm: len(draw_rows[arm])
        for arm in ('226', '233')}, 'intervention_source_counts': source_counts,
    'al1a_psf': {'offsets_um': [0.0], 'weights': [1.0],
        'role': 'same unknown-thickness single-plane default in both arms; weak guard only'},
    'trained': False, 'reserved_dev_cal_final_used': False,
    'public_benchmark_used': False, 'calibrated': False}
(out / 'config.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
with (out / 'cases.jsonl').open('w') as stream:
    for case in cases:
        stream.write(json.dumps({key: case[key] for key in
            ('panel', 'base', 'seed', 'physical_section_id', 'input_sha256')}) + '\n')
with (out / 'attempts.jsonl').open('w') as stream:
    for record in attempts:
        stream.write(json.dumps(record, allow_nan=False) + '\n')

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
al1a_offsets = torch.zeros(1, device='cuda')
al1a_weights = torch.ones(1, device='cuda')
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for label, path in checkpoints:
        arm, step = label.split('_')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['step'] == int(step) and saved['calibrated'] is False
        assert saved['config'] == config[arm]
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for case in cases:
            image = case['image'].cuda()
            offsets, weights = case['offsets'].cuda(), case['weights'].cuda()
            sites, truth, normal = case['sites'].cuda(), case['truth'].cuda(), case['normal'].cuda()
            predicted = model.predict(image)
            prior = (predicted['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-predicted['reflection_logit'][0]),
                F.logsigmoid(predicted['reflection_logit'][0])), -1)).flatten()
            direct_branch = int(prior.argmax())
            states = predicted['state'][0].repeat_interleave(2, 0)
            reflections = torch.arange(2, device='cuda').repeat(model.modes)
            all_rigid = rigid_points_090(states, reflections, sites)
            all_error = (all_rigid - truth[None]).norm(dim=-1).mean(-1) / 1000
            direct_normal = full_frame_state_to_components(states[direct_branch])[1][:, 2]
            direct_angle = torch.rad2deg(torch.acos((direct_normal * normal).sum().abs().clamp_max(1)))
            direct_map = model.map(predicted, offsets,
                torch.tensor([[direct_branch // 2]], device='cuda'),
                torch.tensor([[direct_branch % 2]], device='cuda'),
                (256, 256), context['atlas'], weights, source_shape=(256, 256))
            direct_surface = direct_map['centre_surface_ccf_ap_dv_ml_um'][0, 0].reshape(-1, 3)[
                case['pixels'].cuda()]
            direct_local_mm = (direct_surface - truth).norm(dim=-1).mean() / 1000
            fitted = infer_one_shot(model, image, context['atlas'], offsets, weights)
            final_branch = 2 * fitted['source_selected_component'][0] + fitted['source_selected_component'][1]
            final_state = fitted['state'][0, 0].cuda()
            final_rigid = rigid_points_090(final_state,
                torch.tensor(final_branch % 2, device='cuda'), sites)
            final_normal = full_frame_state_to_components(final_state)[1][:, 2]
            final_surface = fitted['surface'][0, 0].reshape(-1, 3)[case['pixels']].cuda()
            record = {'checkpoint': label, 'panel': case['panel'],
                'physical_section_id': case['physical_section_id'],
                'input_sha256': case['input_sha256'],
                'direct_branch': direct_branch, 'final_branch': final_branch,
                'direct_rigid_mm': float(all_error[direct_branch]),
                'final_rigid_mm': float((final_rigid - truth).norm(dim=-1).mean() / 1000),
                'oracle160_direct_rigid_mm': float(all_error.min()),
                'direct_normal_deg': float(direct_angle),
                'final_normal_deg': float(torch.rad2deg(torch.acos(
                    (final_normal * normal).sum().abs().clamp_max(1)))),
                'direct_local_mm': float(direct_local_mm),
                'final_local_mm': float((final_surface - truth).norm(dim=-1).mean() / 1000)}
            rows.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        for case in al1a_cases:
            image, truth = case['image'].cuda(), case['truth']
            predicted = model.predict(image)
            prior = (predicted['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-predicted['reflection_logit'][0]),
                F.logsigmoid(predicted['reflection_logit'][0])), -1)).flatten()
            direct_branch = int(prior.argmax())
            states = predicted['state'][0].repeat_interleave(2, 0)
            reflections = torch.arange(2, device='cuda').repeat(model.modes)
            mapped = rigid_points_090(states, reflections, five)
            error = (mapped - truth[None]).norm(dim=-1).mean(-1) / 1000
            direct_points = mapped[direct_branch]
            fitted = infer_one_shot(model, image, context['atlas'], al1a_offsets, al1a_weights)
            final_branch = 2 * fitted['source_selected_component'][0] + fitted['source_selected_component'][1]
            final_points = rigid_points_090(fitted['state'][0, 0].cuda(),
                torch.tensor(final_branch % 2, device='cuda'), five)
            target_normal = torch.linalg.cross(truth[1] - truth[0], truth[2] - truth[0])
            direct_normal = torch.linalg.cross(direct_points[1] - direct_points[0],
                direct_points[2] - direct_points[0])
            final_normal = torch.linalg.cross(final_points[1] - final_points[0],
                final_points[2] - final_points[0])
            record = {'checkpoint': label, 'panel': 'AL1A_weak_train',
                **{key: case[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                    'source_section_id', 'source_thumbnail', 'source_thumbnail_sha256')},
                'direct_branch': direct_branch, 'final_branch': final_branch,
                'direct_weak_five_point_mm': float(error[direct_branch]),
                'final_weak_five_point_mm': float((final_points - truth).norm(dim=-1).mean() / 1000),
                'oracle160_direct_weak_five_point_mm': float(error.min()),
                'direct_weak_normal_deg': float(torch.rad2deg(torch.acos(
                    F.cosine_similarity(direct_normal[None], target_normal[None]).abs().clamp_max(1)))[0]),
                'final_weak_normal_deg': float(torch.rad2deg(torch.acos(
                    F.cosine_similarity(final_normal[None], target_normal[None]).abs().clamp_max(1)))[0])}
            rows.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        print(json.dumps({'event': 'checkpoint_scored', 'checkpoint': label,
            'synthetic_cases': len(cases), 'AL1A_weak_sections': len(al1a_cases)}), flush=True)
        torch.cuda.empty_cache()

summary = {}
for label, _ in checkpoints:
    summary[label] = {}
    for panel in ('226_frozen_v7', '233_new_lsfm_v7', 'AL1A_weak_train'):
        group = [row for row in rows if row['checkpoint'] == label and row['panel'] == panel]
        names = (('direct_rigid_mm', 'final_rigid_mm', 'oracle160_direct_rigid_mm',
                  'direct_normal_deg', 'final_normal_deg', 'direct_local_mm', 'final_local_mm')
                 if panel != 'AL1A_weak_train' else
                 ('direct_weak_five_point_mm', 'final_weak_five_point_mm',
                  'oracle160_direct_weak_five_point_mm', 'direct_weak_normal_deg',
                  'final_weak_normal_deg'))
        summary[label][panel] = {'n': len(group),
            **{name: {'mean': float(np.mean([row[name] for row in group])),
                      'median': float(np.median([row[name] for row in group]))}
               for name in names},
            'branch_changes': sum(row['direct_branch'] != row['final_branch'] for row in group)}
        direct = 'direct_rigid_mm' if panel != 'AL1A_weak_train' else 'direct_weak_five_point_mm'
        final = 'final_rigid_mm' if panel != 'AL1A_weak_train' else 'final_weak_five_point_mm'
        summary[label][panel]['median_case_paired_fit_gain_mm'] = float(np.median([
            row[direct] - row[final] for row in group]))

comparisons = {}
for step in (0, 400, 800, 1200):
    control_label, intervention_label = f'226_{step:04d}', f'233_{step:04d}'
    control_rows = [row for row in rows if row['checkpoint'] == control_label]
    intervention_rows = [row for row in rows if row['checkpoint'] == intervention_label]
    assert len(control_rows) == len(intervention_rows) == 180
    assert all(a['panel'] == b['panel'] and
        a.get('physical_section_id', a.get('source_section_id')) ==
        b.get('physical_section_id', b.get('source_section_id'))
        for a, b in zip(control_rows, intervention_rows, strict=True))
    comparisons[str(step)] = {}
    for panel in ('226_frozen_v7', '233_new_lsfm_v7', 'AL1A_weak_train'):
        a = [row for row in control_rows if row['panel'] == panel]
        b = [row for row in intervention_rows if row['panel'] == panel]
        metric = 'final_rigid_mm' if panel != 'AL1A_weak_train' else 'final_weak_five_point_mm'
        comparisons[str(step)][panel] = {
            'median_final_gain_226_minus_233_mm':
                summary[control_label][panel][metric]['median'] -
                summary[intervention_label][panel][metric]['median'],
            'median_case_paired_final_gain_226_minus_233_mm':
                float(np.median([x[metric] - y[metric] for x, y in zip(a, b, strict=True)]))}
    normal_regression = (summary[intervention_label]['226_frozen_v7']['final_normal_deg']['median'] -
                         summary[control_label]['226_frozen_v7']['final_normal_deg']['median'])
    al1a_regression = (summary[intervention_label]['AL1A_weak_train']['final_weak_five_point_mm']['median'] -
                       summary[control_label]['AL1A_weak_train']['final_weak_five_point_mm']['median'])
    comparisons[str(step)]['frozen_normal_regression_deg'] = normal_regression
    comparisons[str(step)]['AL1A_weak_regression_mm'] = al1a_regression
    comparisons[str(step)]['gate_pass'] = bool(matched_draws and
        comparisons[str(step)]['226_frozen_v7']['median_final_gain_226_minus_233_mm'] >= .30 and
        normal_regression <= 2 and al1a_regression <= .25)
summary['matched_draws'] = matched_draws
summary['first_draw_mismatch'] = first_draw_mismatch
summary['intervention_source_counts'] = source_counts
summary['comparisons'] = comparisons
summary['any_gate_pass'] = any(item['gate_pass'] for item in comparisons.values())
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({**manifest, 'row_count': len(rows),
    'synthetic_cases': len(cases), 'AL1A_weak_sections': len(al1a_cases),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'cases.jsonl', 'attempts.jsonl', 'rows.jsonl', 'summary.json')}},
    indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'row_count': len(rows),
    'matched_draws': matched_draws, 'any_gate_pass': summary['any_gate_pass']}), flush=True)
