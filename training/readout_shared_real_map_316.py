"""Prespecified TRAIN-inner readout of the matched 316 map-only and shared-encoder arms."""
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
from scipy.ndimage import map_coordinates

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import (
    physical_ouv_to_frame, quicknii_to_allen_points, quicknii_to_allen_vectors)
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out
from training.global_atlas_contrast_090 import rigid_points_090

SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/SHARED_REAL_MAP_316_PROTOCOL_20261015.md'
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
PANEL = ROOT / 'runs/first_choice_exposure_309_001/panels.pt'
PANEL_DRAWS = PANEL.with_name('panel_draws.jsonl')
PANEL_CONFIG = PANEL.with_name('config.json')
PARENT_314 = ROOT / 'reports/acquired_map_314_AL1A/rows.jsonl'
RUN = ROOT / 'runs/shared_real_map_316_001'
OUT = RUN / 'readout'
CHECKPOINTS = {
    'parent_309': ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt',
    'map_only': RUN / 'map_only/map_step_00600.pt',
    'shared_encoder': RUN / 'shared_encoder/map_step_00600.pt',
}
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert ARCHIVE.stat().st_size == split['archive_bytes']
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

input_paths = {
    'protocol': PROTOCOL, 'split': SPLIT, 'panel': PANEL,
    'panel_draws': PANEL_DRAWS, 'panel_config': PANEL_CONFIG,
    'parent_314_rows': PARENT_314, 'run_config': RUN / 'config.json',
    'run_completed': RUN / 'completed.json',
    **{f'{arm}_{name}': RUN / arm / name
       for arm in ('map_only', 'shared_encoder')
       for name in ('draws.jsonl', 'training.jsonl', 'completed.json')},
    **{f'{name}_checkpoint': path for name, path in CHECKPOINTS.items()},
}
input_sha256 = {}
for name, path in input_paths.items():
    with path.open('rb') as stream:
        input_sha256[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
assert input_sha256['map_only_draws.jsonl'] == input_sha256['shared_encoder_draws.jsonl']
expected_314 = {row['section_id']: row for row in
                (json.loads(line) for line in PARENT_314.read_text(encoding='utf8').splitlines())}
assert len(expected_314) == 52

side, grid = 192, 96
axis = (np.arange(grid) + .5) / grid - .5 / side
y, x = np.meshgrid(axis, axis, indexing='ij')
five = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                     [95.5, 95.5]], device='cuda') / side
flags = torch.arange(2, device='cuda')[None, None]
choice = torch.zeros((1, 1), device='cuda', dtype=torch.long)
real_offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
real_weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
panel_offsets = torch.linspace(-31.25, 31.25, 9, device='cuda')
panel_weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda') / 16
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()

config = {
    'script_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'input_sha256': input_sha256, 'source_doi': split['source_doi'],
    'archive_md5_from_split': split['archive_md5'], 'archive_bytes': split['archive_bytes'],
    'label_role': 'AL1A and first 64 frozen v10 cases are TRAIN-inner regression guards only',
    'al1a': 'exact 314 supplied-pose 96-grid unweighted full-canvas affine-free field RMS',
    'al1a_input': 'max(red,green), p99, four zero channels, 192 pixels',
    'field_sign': 'reference minus observed',
    'permutation': 'cyclic next AL1A input, same original supplied human plane and field target',
    'panel': 'first 64 frozen 309 v10 cases, eight per base 56-63',
    'panel_reflection': 'oracle branch: lower rigid-site error to saved truth under saved exact state',
    'panel_psf': '312 fixed nine-point +/-31.25 um approximation; frozen 309 panel omitted PSF',
    'paired_draws_identical': True, 'calibrated': False, 'public_benchmark_used': False,
}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

real_cases, real_rows = [], []
with ZipFile(ARCHIVE) as source_zip:
    animal = 'AL1A'
    valid = {item.findtext('name') for item in ElementTree.fromstring(
        source_zip.read(f'{animal}/{animal}-info.xml')).findall('slices')
        if item.findtext('valid') == '1'}
    registration = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
    for metadata in registration['slices']:
        section = Path(metadata['filename']).name.removesuffix('-thumb.png')
        if section not in valid:
            continue
        image_name = f'{animal}/{metadata["filename"]}'
        image_bytes = source_zip.read(image_name)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        width, height = image.size
        rgb = np.asarray(image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        inputs = torch.zeros(1, 5, side, side)
        inputs[0, 0] = torch.from_numpy(gray)
        q = torch.tensor(metadata['anchoring'], dtype=torch.float32).reshape(3, 3)
        ouv = torch.stack((quicknii_to_allen_points(q[0]),
            quicknii_to_allen_vectors(q[1]), quicknii_to_allen_vectors(q[2]))) * 25
        state = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).cuda()
        frame = full_frame_state_to_components(state)[1]
        ouv = ouv.cuda()
        human_five = ouv[0] + five[:, :1] * ouv[1] + five[:, 1:] * ouv[2]
        dx_bytes = source_zip.read(f'{animal}/dispFields/{section}-dispFieldX.csv')
        dy_bytes = source_zip.read(f'{animal}/dispFields/{section}-dispFieldY.csv')
        mask_bytes = source_zip.read(f'{animal}/masks/{section}-mask.png')
        dx = np.loadtxt(BytesIO(dx_bytes), delimiter=',')
        dy = np.loadtxt(BytesIO(dy_bytes), delimiter=',')
        query = [y * height * .2 - 1, x * width * .2 - 1]
        sampled_x = map_coordinates(dx, query, order=1, mode='nearest')
        sampled_y = map_coordinates(dy, query, order=1, mode='nearest')
        displacement = (torch.from_numpy(sampled_x).cuda().float()[..., None] * ouv[1] / width
            + torch.from_numpy(sampled_y).cuda().float()[..., None] * ouv[2] / height)
        local = torch.einsum('ij,hwj->ihw', frame.T, displacement)
        target = project_surface_affine_out(local[None])[0][0]
        mask = torch.from_numpy(np.asarray(Image.open(BytesIO(mask_bytes)).convert('L').resize(
            (grid, grid), Image.Resampling.NEAREST)) > 127).cuda()
        zero = float(target[:, mask].square().sum(0).mean().sqrt() / 1000)
        original = expected_314[section]
        assert hashlib.sha256(image_bytes).hexdigest() == original['source_thumbnail_sha256']
        assert hashlib.sha256(dx_bytes).hexdigest() == original['displacement_x_sha256']
        assert hashlib.sha256(dy_bytes).hexdigest() == original['displacement_y_sha256']
        assert hashlib.sha256(mask_bytes).hexdigest() == original['mask_sha256']
        assert abs(zero - original['rms_mm']['zero']) < 1e-5
        real_cases.append({'inputs': inputs, 'state': state.cpu(), 'human_five': human_five.cpu(),
                           'target': target.cpu(), 'mask': mask.cpu()})
        real_rows.append({'animal_id': animal, 'specimen_id': animal, 'experiment_id': None,
            'label_role': 'TRAIN-inner approximate human plane and field',
            'section_id': section, 'source_thumbnail': image_name,
            'source_thumbnail_sha256': original['source_thumbnail_sha256'],
            'displacement_x_sha256': original['displacement_x_sha256'],
            'displacement_y_sha256': original['displacement_y_sha256'],
            'mask_sha256': original['mask_sha256'], 'tissue_sites_96': int(mask.sum()),
            'rms_mm': {'zero': zero}, 'permuted_image_rms_mm': {},
            'blind_first_five_site_gap_mm': {}})
assert len(real_rows) == 52 and {row['section_id'] for row in real_rows} == set(expected_314)
for index, row in enumerate(real_rows):
    donor = real_rows[(index + 1) % len(real_rows)]
    row['permuted_from_section_id'] = donor['section_id']
    row['permuted_from_thumbnail_sha256'] = donor['source_thumbnail_sha256']

panel = torch.load(PANEL, map_location='cpu', weights_only=True)['v10'][:64]
assert len(panel) == 64
panel_receipts = []
with PANEL_DRAWS.open('rb') as stream:
    for line in stream:
        receipt = json.loads(line)
        if receipt['cohort'] == 'v10' and receipt['accepted']:
            panel_receipts.append((receipt, hashlib.sha256(line).hexdigest()))
assert len(panel_receipts) == 256
panel_cases, panel_rows = [], []
for index, case in enumerate(panel):
    meta = case['meta']
    assert meta['base'] == 56 + index % 8
    receipt, receipt_sha256 = panel_receipts[index]
    assert (receipt['slot'], receipt['base'], receipt['variant'], receipt['seed']) == (
        index, meta['base'], meta['variant'], meta['seed'])
    assert receipt['provenance']['physical_section_id'] == meta['physical_section_id']
    chart = case['chart'].cuda()
    truth = case['truth'].cuda()
    state = case['state'].cuda()
    true_states = state[None, None, None].expand(1, 1, 2, 12)
    true_points = rigid_points_090(true_states, flags, chart[None, None, None], (side, side))
    true_gaps = (true_points - truth[None, None, None]).norm(dim=-1).mean(-1)[0, 0] / 1000
    oracle_reflection = int(true_gaps.argmin())
    panel_cases.append({'inputs': case['inputs'], 'chart': case['chart'],
                        'truth': case['truth'], 'state': case['state'],
                        'oracle_reflection': oracle_reflection})
    panel_rows.append({'cohort': 'v10', 'panel_slot': index, 'label_role': 'TRAIN-inner frozen case',
        **meta, 'panel_draw_receipt_sha256': receipt_sha256,
        'input_tensor_sha256': hashlib.sha256(case['inputs'].contiguous().numpy().tobytes()).hexdigest(),
        'chart_tensor_sha256': hashlib.sha256(case['chart'].contiguous().numpy().tobytes()).hexdigest(),
        'truth_tensor_sha256': hashlib.sha256(case['truth'].contiguous().numpy().tobytes()).hexdigest(),
        'state_tensor_sha256': hashlib.sha256(case['state'].contiguous().numpy().tobytes()).hexdigest(),
        'oracle_reflection': oracle_reflection,
        'exact_pose_rigid_mm': float(true_gaps[oracle_reflection]),
        'reflection_branch_gap_mm': float((true_gaps[0] - true_gaps[1]).abs()),
        'exact_pose_map_mm': {}, 'blind_first_rigid_mm': {}})

for name, path in CHECKPOINTS.items():
    model, _ = load_one_shot_checkpoint(CHECKPOINTS['parent_309'], 'cuda')
    if name != 'parent_309':
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == 600 and checkpoint['arm'] == name
        model.load_state_dict(checkpoint['model'])
        del checkpoint
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        for index, case in enumerate(real_cases):
            prediction = model.predict(case['inputs'].cuda())
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(1, model.modes, 2)
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            points = rigid_points_090(states, reflections, five, (side, side))
            gaps = (points - case['human_five'].cuda()[None, None, None]).norm(
                dim=-1).mean(-1).flatten(1) / 1000
            first = mass.argmax(-1)
            real_rows[index]['blind_first_five_site_gap_mm'][name] = float(gaps[0, first[0]])
            supplied = {**prediction, 'state': prediction['state'].clone()}
            supplied['state'][0, 0] = case['state'].cuda()
            field = model.map(supplied, real_offsets, choice, choice, (grid, grid),
                atlas, real_weights, feature_side=grid, source_shape=(side, side))[
                'local_displacement_um'][0, 0]
            mask = case['mask'].cuda()
            target = case['target'].cuda()
            rms = float((field[:, mask] - target[:, mask]).square().sum(0).mean().sqrt() / 1000)
            real_rows[index]['rms_mm'][name] = rms
            if name == 'parent_309':
                assert abs(rms - expected_314[real_rows[index]['section_id']]['rms_mm'][name]) < 1e-5
            permuted = model.predict(real_cases[(index + 1) % len(real_cases)]['inputs'].cuda())
            permuted_supplied = {**permuted, 'state': permuted['state'].clone()}
            permuted_supplied['state'][0, 0] = case['state'].cuda()
            permuted_field = model.map(permuted_supplied, real_offsets, choice, choice,
                (grid, grid), atlas, real_weights, feature_side=grid,
                source_shape=(side, side))['local_displacement_um'][0, 0]
            real_rows[index]['permuted_image_rms_mm'][name] = float((
                permuted_field[:, mask] - target[:, mask]).square().sum(0).mean().sqrt() / 1000)

        for index, case in enumerate(panel_cases):
            inputs = case['inputs'][None].cuda()
            chart = case['chart'].cuda()
            truth = case['truth'].cuda()
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(1, model.modes, 2)
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            points = rigid_points_090(states, reflections, chart[None, None, None], (side, side))
            gaps = (points - truth[None, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            first = mass.argmax(-1)
            panel_rows[index]['blind_first_rigid_mm'][name] = float(gaps[0, first[0]])
            supplied = {**prediction, 'state': prediction['state'].clone()}
            supplied['state'][0, 0] = case['state'].cuda()
            oracle = torch.tensor([[case['oracle_reflection']]], device='cuda')
            mapped = model.map(supplied, panel_offsets, choice, oracle, (side, side),
                               atlas, panel_weights)
            pixels = (chart[:, 1] * side).long() * side + (chart[:, 0] * side).long()
            sites = mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0].flatten(1, 2).gather(
                1, pixels[None, :, None].expand(1, -1, 3))
            panel_rows[index]['exact_pose_map_mm'][name] = float(
                (sites[0] - truth).norm(dim=-1).mean() / 1000)
    print(json.dumps({'event': 'model_readout', 'model': name}), flush=True)
    del model
    torch.cuda.empty_cache()

summary = {'al1a_sections': len(real_rows), 'panel_cases': len(panel_rows),
           'paired_draws_identical': True, 'al1a': {}, 'panel': {},
           'image_dependence': {}, 'predeclared_gate': {},
           'calibrated': False, 'public_benchmark_used': False}
for name in ('zero', *CHECKPOINTS):
    values = np.asarray([row['rms_mm'][name] for row in real_rows])
    summary['al1a'][name] = {'map_rms_mean_mm': float(values.mean()),
        'map_rms_median_mm': float(np.median(values)),
        'map_rms_p90_mm': float(np.quantile(values, .9))}
    if name == 'zero':
        continue
    pose = np.asarray([row['blind_first_five_site_gap_mm'][name] for row in real_rows])
    permuted = np.asarray([row['permuted_image_rms_mm'][name] for row in real_rows])
    summary['al1a'][name]['blind_first_five_site_gap_mean_mm'] = float(pose.mean())
    summary['al1a'][name]['worse_than_zero_by_gt_0p1mm'] = int(sum(
        row['rms_mm'][name] - row['rms_mm']['zero'] > .1 for row in real_rows))
    summary['image_dependence'][name] = {
        'permuted_map_rms_mean_mm': float(permuted.mean()),
        'original_minus_permuted_mean_mm': float(values.mean() - permuted.mean()),
        'permuted_minus_original_mean_mm': float(permuted.mean() - values.mean()),
        'original_better_sections': int((values < permuted).sum())}
    exact = np.asarray([row['exact_pose_map_mm'][name] for row in panel_rows])
    blind = np.asarray([row['blind_first_rigid_mm'][name] for row in panel_rows])
    summary['panel'][name] = {'exact_pose_map_mean_mm': float(exact.mean()),
        'blind_first_rigid_mean_mm': float(blind.mean()),
        'blind_first_gt5mm': int((blind > 5).sum())}

for name in ('map_only', 'shared_encoder'):
    real = summary['al1a'][name]
    panel_score = summary['panel'][name]
    parent_real = summary['al1a']['parent_309']
    parent_panel = summary['panel']['parent_309']
    image = summary['image_dependence'][name]
    tests = {
        'al1a_vs_zero_0p9': real['map_rms_mean_mm'] <= .9 * summary['al1a']['zero']['map_rms_mean_mm'],
        'al1a_vs_parent_0p9': real['map_rms_mean_mm'] <= .9 * parent_real['map_rms_mean_mm'],
        'al1a_worse_zero_at_most_four': real['worse_than_zero_by_gt_0p1mm'] <= 4,
        'panel_exact_map_worsens_at_most_10pct': panel_score['exact_pose_map_mean_mm']
            <= 1.1 * parent_panel['exact_pose_map_mean_mm'],
        'panel_blind_mean_worsens_at_most_0p15mm': panel_score['blind_first_rigid_mean_mm']
            <= parent_panel['blind_first_rigid_mean_mm'] + .15,
        'panel_gt5mm_at_most_two_extra': panel_score['blind_first_gt5mm']
            <= parent_panel['blind_first_gt5mm'] + 2,
        'al1a_blind_mean_worsens_at_most_0p20mm': real['blind_first_five_site_gap_mean_mm']
            <= parent_real['blind_first_five_site_gap_mean_mm'] + .20,
        'permuted_worse_by_at_least_0p005mm': image['permuted_minus_original_mean_mm'] >= .005,
        'original_better_on_at_least_35': image['original_better_sections'] >= 35,
    }
    summary['predeclared_gate'][name] = {'tests': tests, 'promising': bool(all(tests.values()))}
summary['predeclared_gate']['shared_encoder']['beats_map_only_by_at_least_0p005mm'] = bool(
    summary['al1a']['map_only']['map_rms_mean_mm']
    - summary['al1a']['shared_encoder']['map_rms_mean_mm'] >= .005)
summary['predeclared_gate']['shared_encoder']['shared_encoder_attribution'] = bool(
    summary['predeclared_gate']['shared_encoder']['promising']
    and summary['predeclared_gate']['shared_encoder']['beats_map_only_by_at_least_0p005mm'])

(OUT / 'al1a_rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n'
    for row in real_rows), encoding='utf8')
(OUT / 'panel_rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n'
    for row in panel_rows), encoding='utf8')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
output_sha256 = {}
for name in ('config.json', 'al1a_rows.jsonl', 'panel_rows.jsonl', 'summary.json'):
    with (OUT / name).open('rb') as stream:
        output_sha256[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(OUT / 'completed.json').write_text(json.dumps({'al1a_sections': len(real_rows),
    'panel_cases': len(panel_rows), 'script_sha256': config['script_sha256'],
    'protocol_sha256': input_sha256['protocol'], 'paired_draws_identical': True,
    'output_sha256': output_sha256, 'calibrated': False,
    'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
