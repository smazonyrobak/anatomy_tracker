"""Paired real-fluorescence global matcher, with AL1A excluded from gradients."""

import copy
import hashlib
import json
import os
import sys
import time
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
protocol = source.parents[1] / 'docs/publication/LUPORI_CROSSMODAL_MATCHER_191_INNER_PROTOCOL_20261011.md'
split_path = source.with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
archive = root / 'data/lupori_joint_185/rawData.zip'
parent_run = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A'
parent = parent_run / 'joint_step_00800.pt'
run = root / 'runs/lupori_crossmodal_matcher_191_inner_AL1A_retry1'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert sha(parent) == json.loads((parent_run / 'completed.json').read_text())[
    'checkpoint_sha256']['800']
assert archive.stat().st_size == split['archive_bytes']
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
seed, updates, side = 20261011191, 600, 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 800 and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved
model.requires_grad_(False)
heads = {arm: copy.deepcopy(model.global_plane_matcher_120).cuda().train().requires_grad_(True)
         for arm in ('full', 'support_only')}
assert all(torch.equal(a, b) for a, b in zip(
    heads['full'].state_dict().values(), heads['support_only'].state_dict().values()))
optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
              for arm, head in heads.items()}

records, images, targets = [], [], []
with ZipFile(archive) as source_zip:
    for animal in split['train']:
        if animal == 'AL1A':
            continue
        info = ElementTree.fromstring(source_zip.read(f'{animal}/{animal}-info.xml'))
        valid = {row.findtext('name') for row in info.findall('slices')
                 if row.findtext('valid') == '1'}
        visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        assert visualign['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        for row in visualign['slices']:
            section_id = Path(row['filename']).name.removesuffix('-thumb.png')
            if section_id not in valid:
                continue
            image_bytes = source_zip.read(f'{animal}/{row["filename"]}')
            rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
                (side, side), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
            q = torch.tensor(row['anchoring'], dtype=torch.float32).reshape(3, 3)
            ouv = torch.stack((quicknii_to_allen_points(q[0]),
                quicknii_to_allen_vectors(q[1]), quicknii_to_allen_vectors(q[2]))) * 25
            images.append(gray)
            targets.append(ouv)
            records.append({'animal_id': animal, 'specimen_id': animal,
                'experiment_id': None, 'section_id': section_id,
                'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest()})
assert len(records) == 188 and len({(r['animal_id'], r['section_id']) for r in records}) == 188
images = torch.from_numpy(np.stack(images)).cuda()
targets = torch.stack(targets).cuda()
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
source_names = ('train_lupori_crossmodal_matcher_191.py',
    'arbitrary_plane_one_shot_model.py', 'global_plane_matcher_120.py',
    'global_atlas_contrast_090.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_allen_atlas_binding_v6.py')
config = {'seed': seed, 'updates': updates, 'side': side, 'train_sections': len(records),
    'train_animals': ['CC2B', 'CC4B', 'CC4A'], 'guard_animal': 'AL1A',
    'protocol_sha256': sha(protocol), 'split_sha256': sha(split_path),
    'archive_md5': split['archive_md5'], 'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'source_sha256': {name: sha(source.parent / name) for name in source_names},
    'input': 'whole RGB 256, max(red,green)/255, image-wide p99, four channels zero',
    'loss': 'four blind candidates, softmax(-fivepoint mm/.7) cross entropy',
    'optimized': 'global_plane_matcher_120 only, independent full/support_only arms',
    'atlas_offsets_um': offsets.tolist(), 'atlas_weights': weights.tolist(),
    'checkpoints': [0, 300, 600], 'external_pretrained_weights_used': False,
    'calibrated': False, 'public_benchmark_used': False,
    'development_calibration_final_read': False}
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    path = run / f'matcher_step_{step:05d}.pt'
    temp = path.with_suffix('.tmp')
    torch.save({'step': step, 'heads': {arm: head.state_dict()
        for arm, head in heads.items()}, 'optimizers': {arm: opt.state_dict()
        for arm, opt in optimizers.items()}, 'numpy_rng': rng.bit_generator.state,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'config': config, 'calibrated': False}, temp)
    os.replace(temp, path)


save(0)
order = rng.permutation(len(records))
cursor, epoch = 0, 0
started = time.perf_counter()
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as training:
    for step in range(1, updates + 1):
        if cursor == len(order):
            order = rng.permutation(len(records))
            cursor = 0
            epoch += 1
        index = int(order[cursor])
        cursor += 1
        inputs = torch.zeros(1, 5, side, side, device='cuda')
        inputs[0, 0] = images[index]
        truth = targets[index, 0] + five[:, :1] * targets[index, 1] + five[:, 1:] * targets[index, 2]
        with torch.no_grad():
            prediction = model.predict(inputs)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            ids = prior.topk(16).indices
            states = prediction['state'][0, ids // 2]
            flags = ids.remainder(2)
            error = (rigid_points_090(states, flags, five) - truth[None]
                ).norm(dim=-1).mean(-1) / 1000
            best = int(error.argmin())
            wrong = (error >= error[best] + 1.5).nonzero()[:, 0]
            hard = int(wrong[0]) if len(wrong) else int(error.argmax())
            selected = list(dict.fromkeys((best, 0, hard)))
            for candidate in error.argsort():
                if len(selected) == 4:
                    break
                if int(candidate) not in selected:
                    selected.append(int(candidate))
            selected = torch.tensor(selected, device='cuda')
            target = F.softmax(-error[selected] / .7, -1).detach()
        draws.write(json.dumps({'update': step, 'epoch': epoch, **records[index],
            'branch_ids': ids[selected].tolist(),
            'five_point_mm': error[selected].tolist()}, allow_nan=False) + '\n')
        measures = {'update': step, 'epoch': epoch, 'animal_id': records[index]['animal_id'],
            'prior_top_mm': float(error[0]), 'best16_mm': float(error.min())}
        for arm, head in heads.items():
            result = global_plane_match(model, prediction, (ids[selected] // 2)[None],
                (ids[selected] % 2)[None], offsets, weights, atlas, (side, side),
                side=24, support_only=arm == 'support_only', matcher=head)
            score = result['score'][0]
            loss = -(target * F.log_softmax(score, -1)).sum()
            optimizers[arm].zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizers[arm].step()
            measures[arm] = {'loss': float(loss.detach()),
                'gradient_norm': float(gradient.detach()),
                'selected_mm': float(error[selected[score.detach().argmax()]])}
        training.write(json.dumps(measures, allow_nan=False) + '\n')
        if step in (300, 600):
            draws.flush()
            training.flush()
            save(step)
            print(json.dumps({'event': 'checkpoint', 'step': step,
                'elapsed_seconds': round(time.perf_counter() - started, 1),
                'last_loss_full': measures['full']['loss'],
                'last_loss_support_only': measures['support_only']['loss']}), flush=True)

done = {'updates': updates, 'gradient_animals': ['CC2B', 'CC4B', 'CC4A'],
    'gradient_sections': len(records), 'guard_animal_used_in_training': False,
    'source_sha256': config['source_sha256'], 'protocol_sha256': config['protocol_sha256'],
    'config_sha256': sha(run / 'config.json'), 'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'matcher_step_{step:05d}.pt')
        for step in (0, 300, 600)}, 'calibrated': False,
    'public_benchmark_used': False, 'external_pretrained_weights_used': False}
(run / 'completed.json').write_text(json.dumps(done, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
    'elapsed_seconds': round(time.perf_counter() - started, 1)}), flush=True)
