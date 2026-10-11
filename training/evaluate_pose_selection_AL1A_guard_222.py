"""Reserved TRAIN-animal human-VisuAlign guard for both 222 arms."""

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
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
protocol = source.parents[1] / 'docs/publication/POSE_SELECTION_222_PROTOCOL_20261011.md'
out = root / 'runs/pose_selection_222_AL1A_train_guard_001'
parent = root / 'runs/pose_framing_221_framed_train_001/joint_step_00400.pt'
arms = {arm: root / f'runs/pose_selection_222_{arm}_train_001'
        for arm in ('control', 'intervention')}
assert source.drive.upper() == root.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists() and archive.stat().st_size == split['archive_bytes']
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent_done = json.loads((parent.parent / 'completed.json').read_text())
parent_config = json.loads((parent.parent / 'config.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['400']
assert sha(parent.parent / 'config.json') == parent_done['config_sha256']
checkpoints = {'221_framed_400': parent}
configs = {}
for arm, directory in arms.items():
    done = json.loads((directory / 'completed.json').read_text())
    configs[arm] = json.loads((directory / 'config.json').read_text())
    assert done['updates'] == 400 and done['parent_checkpoint_sha256'] == sha(parent)
    assert done['protocol_sha256'] == sha(protocol)
    assert sha(directory / 'config.json') == done['config_sha256']
    assert done['calibrated'] is False and done['public_benchmark_used'] is False
    assert configs[arm]['inner_guard_animal'] == 'AL1A'
    assert 'AL1A' not in configs[arm]['lupori_gradient_animals']
    for step in (0, 200, 400):
        checkpoints[f'222_{arm}_{step}'] = directory / f'joint_step_{step:05d}.pt'
checkpoint_sha256 = {label: sha(path) for label, path in checkpoints.items()}
assert all(checkpoint_sha256[f'222_{arm}_{step}'] ==
           json.loads((directory / 'completed.json').read_text())['checkpoint_sha256'][str(step)]
           for arm, directory in arms.items() for step in (0, 200, 400))

five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / 256
images, targets, cases = [], [], []
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
        o, u, v = torch.tensor(row['anchoring'], dtype=torch.float32).reshape(3, 3)
        quicknii = o + five.cpu()[:, :1] * u + five.cpu()[:, 1:] * v
        images.append(gray)
        targets.append(quicknii_to_allen_points(quicknii).numpy() * 25)
        cases.append({'animal_id': 'AL1A', 'specimen_id': 'AL1A',
            'experiment_id': None, 'source_section_id': section_id,
            'source_thumbnail': thumbnail,
            'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest()})
assert len(cases) == 52
images = np.stack(images)
targets = torch.from_numpy(np.stack(targets)).cuda()

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
flags = torch.arange(2, device='cuda')[None, None]
rows = []
with torch.inference_mode():
    for label, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        expected_step = 400 if label == '221_framed_400' else int(label.rsplit('_', 1)[-1])
        assert saved['step'] == expected_step and saved['calibrated'] is False
        assert saved['config'] == (parent_config if label == '221_framed_400'
                                   else configs[label.split('_')[1]])
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for start in range(0, len(cases), 8):
            batch = images[start:start + 8]
            inputs = torch.zeros(len(batch), 5, 256, 256, device='cuda')
            inputs[:, 0] = torch.from_numpy(batch).cuda()
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(batch), model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            mapped = rigid_points_090(states, reflections, five)
            errors = (mapped - targets[start:start + len(batch), None, None]
                     ).norm(dim=-1).mean(-1).flatten(1) / 1000
            chosen = prior.argmax(-1)
            top16 = prior.topk(16, -1).indices
            for j in range(len(batch)):
                selected = mapped.flatten(1, 2)[j, chosen[j]]
                target = targets[start + j]
                selected_normal = torch.linalg.cross(selected[1] - selected[0],
                    selected[2] - selected[0])
                target_normal = torch.linalg.cross(target[1] - target[0],
                    target[2] - target[0])
                cosine = F.cosine_similarity(selected_normal[None],
                    target_normal[None]).abs().clamp_max(1)
                rows.append({'checkpoint': label,
                    'checkpoint_sha256': checkpoint_sha256[label],
                    **cases[start + j], 'top1_branch_id': int(chosen[j]),
                    'top1_weak_five_point_mm': float(errors[j, chosen[j]]),
                    'top1_weak_center_mm': float((selected[4] - target[4]).norm() / 1000),
                    'top1_weak_axis_deg': float(torch.rad2deg(torch.acos(cosine))[0]),
                    'top16_best_weak_five_point_mm': float(errors[j, top16[j]].min()),
                    'best160_weak_five_point_mm': float(errors[j].min())})
        print(json.dumps({'event': 'AL1A_guard', 'checkpoint': label,
            'sections': len(cases)}), flush=True)
for arm in arms:
    zero = [row for row in rows if row['checkpoint'] == f'222_{arm}_0']
    baseline = [row for row in rows if row['checkpoint'] == '221_framed_400']
    assert all(a['source_section_id'] == b['source_section_id'] and
        abs(a['top1_weak_five_point_mm'] - b['top1_weak_five_point_mm']) < 1e-5
        for a, b in zip(zero, baseline, strict=True))

summary = {}
for label in checkpoints:
    group = [row for row in rows if row['checkpoint'] == label]
    values = np.asarray([row['top1_weak_five_point_mm'] for row in group])
    summary[label] = {'sections': len(group),
        'top1_median_mm': float(np.median(values)),
        'top1_mean_mm': float(values.mean()),
        'top1_p90_mm': float(np.percentile(values, 90)),
        'top16_best_median_mm': float(np.median([
            row['top16_best_weak_five_point_mm'] for row in group])),
        'best160_median_mm': float(np.median([
            row['best160_weak_five_point_mm'] for row in group]))}
for arm in arms:
    label = f'222_{arm}_400'
    summary[label]['top1_median_minus_parent_mm'] = (
        summary[label]['top1_median_mm'] - summary['221_framed_400']['top1_median_mm'])
summary['222_intervention_400']['top1_median_minus_control_mm'] = (
    summary['222_intervention_400']['top1_median_mm'] -
    summary['222_control_400']['top1_median_mm'])

out.mkdir(parents=True, exist_ok=False)
config = {'evaluator_source_sha256': sha(source), 'protocol_sha256': sha(protocol),
    'split_sha256': sha(split_path), 'source_doi': split['source_doi'],
    'archive_bytes': split['archive_bytes'], 'archive_md5': split['archive_md5'],
    'AL1A_info_sha256': hashlib.sha256(info_bytes).hexdigest(),
    'AL1A_visualign_sha256': hashlib.sha256(visualign_bytes).hexdigest(),
    'checkpoint_sha256': checkpoint_sha256,
    'training_completion_sha256': {arm: sha(directory / 'completed.json')
        for arm, directory in arms.items()},
    'input': 'whole RGB thumbnail resized 256, max(red,green)/255, image-wide p99 normalized, other channels zero',
    'label_role': 'reserved TRAIN animal human VisuAlign weak reference, not independent ground truth',
    'development_calibration_final_used': False,
    'public_benchmark_used': False, 'calibrated': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({'sections': len(cases), 'rows': len(rows),
    'output_sha256': {name: sha(out / name)
        for name in ('config.json', 'summary.json', 'rows.jsonl')},
    'development_calibration_final_used': False,
    'public_benchmark_used': False, 'calibrated': False}, indent=2, allow_nan=False) + '\n')
print(json.dumps({'event': 'AL1A_guard_complete', 'sections': len(cases),
                  'rows': len(rows)}), flush=True)
