"""Fixed post-training retention readout for the 212 native-anatomy comparison."""

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.arbitrary_plane_geometry import quicknii_to_allen_points
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/ALLEN_NATIVE_STPT_212_MATCHED_PROTOCOL_20261011.md'
train = root / 'runs/allen_native_matched_212_001'
out = root / 'runs/allen_native_matched_212_retention_214_001'
panels = {'v3': root / 'data/joint_in_path_correspondence_128_dev_panel',
          'v4': root / 'data/fresh_v4_pose_dev_panel_132'}
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
archive = root / 'data/lupori_joint_185/rawData.zip'
arms, steps, side = ('control', 'native'), (0, 200, 400), 256
assert source.drive.upper() == root.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
assert all((train / arm / 'completed.json').is_file() for arm in arms)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

completion = {arm: json.loads((train / arm / 'completed.json').read_text()) for arm in arms}
checkpoints = {(arm, step): train / arm / f'joint_step_{step:05d}.pt'
               for arm in arms for step in steps}
assert all(sha(path) == completion[arm]['checkpoint_sha256'][str(step)]
           for (arm, step), path in checkpoints.items())
assert all(completion[arm]['calibrated'] is False and
           completion[arm]['public_benchmark_used'] is False for arm in arms)

panel_rows = {name: [json.loads(line) for line in (path / 'records.jsonl').open()]
              for name, path in panels.items()}
assert all(sha(path / 'records.jsonl') ==
           json.loads((path / 'completed.json').read_text())['records_sha256']
           for path in panels.values())
coronal_rows = [row for row in map(json.loads, (coronal / 'records.jsonl').open())
                if row['training_split'] == 'development']
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
sagittal_rows = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')
assert len(coronal_rows) == 64 and len({row['animal_id'] for row in coronal_rows}) == 6
assert len(sagittal_rows) == 158 and len({row['donor_id'] for row in sagittal_rows}) == 8

five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
axis = torch.arange(side, device='cuda') / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)

al1a_images, al1a_targets, al1a_ids = [], [], []
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
        rgb = np.asarray(Image.open(BytesIO(source_zip.read(
            f'AL1A/{row["filename"]}'))).convert('RGB').resize(
                (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        anchoring = torch.tensor(row['anchoring'], dtype=torch.float32).reshape(3, 3)
        points = anchoring[0] + five.cpu()[:, :1] * anchoring[1] + five.cpu()[:, 1:] * anchoring[2]
        al1a_images.append(gray)
        al1a_targets.append(quicknii_to_allen_points(points).numpy() * 25)
        al1a_ids.append(section_id)
assert len(al1a_ids) == 52

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
rows = []
with torch.inference_mode():
    for arm in arms:
        for step in steps:
            saved = torch.load(checkpoints[arm, step], map_location='cpu', weights_only=True)
            assert saved['step'] == step and saved['arm'] == arm and saved['calibrated'] is False
            model.load_state_dict(saved['model'], strict=True)
            del saved
            for section_id, gray, target in zip(al1a_ids, al1a_images, al1a_targets):
                inputs = torch.zeros(1, 5, side, side, device='cuda')
                inputs[0, 0] = torch.from_numpy(gray).cuda()
                prediction = model.predict(inputs)
                log_flip = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                        F.logsigmoid(prediction['reflection_logit'])), -1)
                branch = int((prediction['log_mass'][..., None] + log_flip).flatten(1).argmax())
                proposed = rigid_points_090(prediction['state'][:, branch // 2],
                    torch.tensor([branch % 2], device='cuda'), five)[0]
                rows.append({'arm': arm, 'step': step, 'cohort': 'lupori_AL1A',
                             'donor_id': 'AL1A', 'section_id': section_id,
                             'error_mm': float((proposed - torch.from_numpy(target).cuda()
                                                ).norm(dim=-1).mean() / 1000)})
            for cohort, panel in panels.items():
                for record in (row for row in panel_rows[cohort] if row['eligible']):
                    with np.load(panel / record['file'], allow_pickle=False) as arrays:
                        inputs = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                        reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
                        valid = torch.from_numpy(arrays['valid_mask'].flatten().copy()).cuda().bool()
                    prediction = model.predict(inputs)
                    log_flip = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                            F.logsigmoid(prediction['reflection_logit'])), -1)
                    branch = int((prediction['log_mass'][..., None] + log_flip).flatten(1).argmax())
                    proposed = rigid_points_090(prediction['state'][:, branch // 2],
                        torch.tensor([branch % 2], device='cuda'), chart)[0]
                    target = rigid_points_090(truth, reflection, chart)[0]
                    rows.append({'arm': arm, 'step': step, 'cohort': f'synthetic_{cohort}',
                                 'donor_id': record['animal_id'],
                                 'section_id': record['section_id'],
                                 'subject_plan_id': record['synthetic_subject_plan_id'],
                                 'error_mm': float((proposed - target).norm(dim=-1)[valid].mean() / 1000)})
            for cohort, items in (('old_coronal', coronal_rows), ('old_sagittal', sagittal_rows)):
                for record in items:
                    index = record['array_row_index']
                    if cohort == 'old_coronal':
                        image = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                            np.zeros((4, 192, 192), np.float32)))[None]
                        inputs = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                               mode='bilinear', align_corners=False)
                        old_pixels = (five * side + .5) * (192 / side) - .5
                        affine = torch.as_tensor(coronal_affines[index], device='cuda', dtype=torch.float32)
                        reference = affine[:, 2] + (old_pixels[:, :1] * affine[:, 0]
                                                      + old_pixels[:, 1:] * affine[:, 1])
                        donor = record['animal_id']
                    else:
                        image = np.concatenate((np.asarray(sagittal_images[index], dtype=np.float32),
                            np.zeros((4, side, side), np.float32)))[None]
                        inputs = torch.from_numpy(image).cuda()
                        affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                                                 device='cuda', dtype=torch.float32)
                        reference = affine[:, 2] + ((five * side)[:, :1] * affine[:, 0]
                                                      + (five * side)[:, 1:] * affine[:, 1])
                        donor = record['donor_id']
                    prediction = model.predict(inputs)
                    log_flip = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                            F.logsigmoid(prediction['reflection_logit'])), -1)
                    branch = int((prediction['log_mass'][..., None] + log_flip).flatten(1).argmax())
                    proposed = rigid_points_090(prediction['state'][:, branch // 2],
                        torch.tensor([branch % 2], device='cuda'), five)[0]
                    rows.append({'arm': arm, 'step': step, 'cohort': cohort,
                                 'donor_id': donor, 'section_id': record['section_id'],
                                 'error_mm': float((proposed - reference).norm(dim=-1).mean() / 1000)})
            print(json.dumps({'arm': arm, 'step': step, 'retention_rows': len(rows)}), flush=True)

summary = {}
for cohort in ('lupori_AL1A', 'synthetic_v3', 'synthetic_v4', 'old_coronal', 'old_sagittal'):
    summary[cohort] = {}
    for step in steps:
        paired = {}
        for arm in arms:
            subset = [row for row in rows if row['cohort'] == cohort and row['step'] == step
                      and row['arm'] == arm]
            values = np.array([row['error_mm'] for row in subset])
            donors = sorted({str(row['donor_id']) for row in subset})
            donor_means = {donor: float(np.mean([row['error_mm'] for row in subset
                                                 if str(row['donor_id']) == donor])) for donor in donors}
            report = {
                'sections': len(subset), 'mean_mm': float(values.mean()),
                'median_mm': float(np.median(values)),
                'p90_mm': float(np.percentile(values, 90)),
                'donor_equal_mean_mm': float(np.mean(list(donor_means.values()))),
                'by_donor_mean_mm': donor_means}
            if cohort.startswith('synthetic_'):
                plans = sorted({row['subject_plan_id'] for row in subset})
                report['plan_equal_mean_mm'] = float(np.mean([
                    np.mean([row['error_mm'] for row in subset
                             if row['subject_plan_id'] == plan]) for plan in plans]))
            summary[cohort].setdefault(str(step), {})[arm] = report
            paired[arm] = {row['section_id']: row['error_mm'] for row in subset}
        assert paired['control'].keys() == paired['native'].keys()
        delta = np.array([paired['native'][key] - paired['control'][key]
                          for key in sorted(paired['control'])])
        summary[cohort][str(step)]['native_minus_control'] = {
            'section_mean_delta_mm': float(delta.mean()),
            'section_median_delta_mm': float(np.median(delta))}
assert all(abs(summary[cohort]['0']['native_minus_control']['section_mean_delta_mm']) < 1e-5
           for cohort in summary)

out.mkdir(parents=True, exist_ok=False)
config = {'evaluator_source_sha256': sha(source), 'protocol_sha256': sha(protocol),
          'checkpoint_sha256': {f'{arm}_{step}': sha(path)
                                for (arm, step), path in checkpoints.items()},
          'training_completion_sha256': {arm: sha(train / arm / 'completed.json') for arm in arms},
          'synthetic_panel_records_sha256': {name: sha(path / 'records.jsonl')
                                             for name, path in panels.items()},
          'lupori_archive_sha256': sha(archive),
          'coronal_completed_sha256': sha(coronal / 'completed.json'),
          'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
          'scope': 'fixed internal retention readout; weak real labels and atlas-derived synthetic plans, not expert truth',
          'calibrated': False, 'public_benchmark_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'cohorts': list(summary),
    'output_sha256': {name: sha(out / name) for name in ('config.json', 'summary.json', 'rows.jsonl')},
    'calibrated': False, 'public_benchmark_used': False, 'final_animals_used': False},
    indent=2, allow_nan=False) + '\n')
print(json.dumps({'event': 'retention_readout_complete', 'rows': len(rows)}), flush=True)
