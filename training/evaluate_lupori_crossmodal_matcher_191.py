"""Frozen AL1A animal guard for the paired 191 cross-modality matchers."""

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
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_geometry import quicknii_to_allen_points, quicknii_to_allen_vectors
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
parent = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt'
train = root / 'runs/lupori_crossmodal_matcher_191_inner_AL1A_retry1'
checkpoint = train / 'matcher_step_00600.pt'
out = root / 'runs/lupori_crossmodal_matcher_191_guard_AL1A'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
done = json.loads((train / 'completed.json').read_text())
assert done['updates'] == 600 and not done['guard_animal_used_in_training']
assert sha(checkpoint) == done['checkpoint_sha256']['600']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)
del saved
matchers = {}
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 600 and saved['config'] == json.loads((train / 'config.json').read_text())
for arm in ('full', 'support_only'):
    matchers[arm] = model.global_plane_matcher_120
    if arm == 'support_only':
        import copy
        matchers[arm] = copy.deepcopy(matchers['full'])
    matchers[arm].load_state_dict(saved['heads'][arm], strict=True)
    matchers[arm].eval().requires_grad_(False)
del saved

side = 256
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
animal = 'AL1A'
rows = []
out.mkdir(parents=True, exist_ok=False)
config = {'source_sha256': sha(source), 'train_completion_sha256': sha(train / 'completed.json'),
    'checkpoint_sha256': sha(checkpoint), 'parent_checkpoint_sha256': sha(parent),
    'split_sha256': sha(split_path), 'source_doi': split['source_doi'],
    'animal_id': animal, 'label_role': 'inner TRAIN-animal approximate human plane',
    'input': 'whole RGB 256, max(red,green)/255, image-wide p99, four channels zero',
    'beam': 'first 16 direct pose/reflection proposals', 'evaluated_step': 600,
    'public_benchmark_used': False, 'calibrated': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
with ZipFile(archive) as source_zip, (out / 'rows.jsonl').open('w') as stream:
    info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
    valid = {row.findtext('name') for row in info.findall('slices')
             if row.findtext('valid') == '1'}
    visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
    for metadata in visualign['slices']:
        section_id = Path(metadata['filename']).name.removesuffix('-thumb.png')
        if section_id not in valid:
            continue
        image_bytes = source_zip.read(f'{animal}/{metadata["filename"]}')
        rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
            (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        inputs = torch.zeros(1, 5, side, side, device='cuda')
        inputs[0, 0] = torch.from_numpy(gray).cuda()
        q = torch.tensor(metadata['anchoring'], dtype=torch.float32,
                         device='cuda').reshape(3, 3)
        o = quicknii_to_allen_points(q[0]) * 25
        u = quicknii_to_allen_vectors(q[1]) * 25
        v = quicknii_to_allen_vectors(q[2]) * 25
        truth = o + five[:, :1] * u + five[:, 1:] * v
        with torch.inference_mode():
            prediction = model.predict(inputs)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            ids = prior.topk(16).indices
            states = prediction['state'][0, ids // 2]
            flags = ids.remainder(2)
            error = (rigid_points_090(states, flags, five) - truth[None]
                ).norm(dim=-1).mean(-1) / 1000
            scores = {}
            for arm in ('full', 'support_only'):
                chunks = []
                for start in range(0, 16, 4):
                    result = global_plane_match(model, prediction,
                        (ids[start:start + 4] // 2)[None],
                        (ids[start:start + 4] % 2)[None], offsets, weights,
                        atlas, (side, side), side=24,
                        support_only=arm == 'support_only', matcher=matchers[arm])
                    chunks.append(result['score'][0])
                scores[arm] = torch.cat(chunks)
            choices = {arm: int(score.argmax()) for arm, score in scores.items()}
        row = {'animal_id': animal, 'specimen_id': animal, 'experiment_id': None,
            'section_id': section_id, 'source_thumbnail': f'{animal}/{metadata["filename"]}',
            'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
            'branch_ids': ids.tolist(), 'five_point_mm': error.tolist(),
            'prior_log_mass': prior[ids].tolist(),
            'score_full': scores['full'].tolist(),
            'score_support_only': scores['support_only'].tolist(),
            'choice_full': choices['full'], 'choice_support_only': choices['support_only'],
            'prior_selected_mm': float(error[0]),
            'oracle_16_mm': float(error.min()),
            'full_selected_mm': float(error[choices['full']]),
            'support_only_selected_mm': float(error[choices['support_only']])}
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if len(rows) % 13 == 0:
            stream.flush()
            print(json.dumps({'event': 'guard_milestone', 'sections': len(rows)}), flush=True)

assert len(rows) == 52
summary = {'sections': len(rows), 'oracle_16_le_1p5': sum(
    row['oracle_16_mm'] <= 1.5 for row in rows), 'methods': {}}
for name in ('prior', 'full', 'support_only'):
    values = np.asarray([row[f'{name}_selected_mm'] for row in rows])
    summary['methods'][name] = {'mean_mm': float(values.mean()),
        'median_mm': float(np.median(values)), 'le_1p5': int((values <= 1.5).sum()),
        'gt_5': int((values > 5).sum())}
p = summary['methods']['prior']
f = summary['methods']['full']
s = summary['methods']['support_only']
summary['predeclared_gate'] = {'passed': bool(p['mean_mm'] - f['mean_mm'] >= .15 and
    s['mean_mm'] - f['mean_mm'] >= .15 and f['le_1p5'] >= p['le_1p5'] and
    f['gt_5'] <= p['gt_5'])}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'sections': len(rows),
    'source_sha256': sha(source), 'train_completion_sha256': sha(train / 'completed.json'),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'rows.jsonl', 'summary.json')},
    'public_benchmark_used': False, 'calibrated': False}, indent=2))
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
