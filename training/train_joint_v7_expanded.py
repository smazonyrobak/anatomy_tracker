"""Continue the whole v7 model on expanded synthetic data and weak real labels.

192px, unchanged deployment architecture, joint anatomical feedback. Not a
benchmark, calibration experiment, or constraint-qualified training stage.
Run only after both new dataset completion records have been written.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True

import hashlib
import json
import math
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components
from training.arbitrary_plane_geometry import physical_ouv_to_frame, frame_to_physical_ouv
from training.arbitrary_plane_joint_model_v7 import JointSliceModel
from training.arbitrary_plane_joint_uncertainty import local_rotation_log

RUN = ROOT / 'runs/joint_v7_expanded_joint_001'
PARENT = ROOT / 'runs/joint_v7_direct_joint_001'
SYNTHETIC = ROOT / 'data/joint_v7_training_data_001'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
OLD = ROOT / 'data/joint_v6_coherent_subject_cohort_sections_002'
SEED, STEPS, BATCH, SIDE = 2026093008, 24000, 8, 192
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((PARENT / 'completed.json').read_text())['steps'] == 6000
synthetic_completed = json.loads((SYNTHETIC / 'completed.json').read_text())
real_completed = json.loads((REAL / 'completed.json').read_text())
assert synthetic_completed['section_count'] == 4096 and len(synthetic_completed['shards']) == 8
assert real_completed['training_images'] == 1280 and real_completed['development_images'] == 64
RUN.mkdir(parents=True, exist_ok=False)
config = {
    'seed': SEED, 'stage_updates': STEPS, 'batch': BATCH, 'resolution': [SIDE, SIDE], 'modes': 8, 'fitter_steps': 3,
    'parent_checkpoint': str(PARENT / 'joint_step_06000.pt'),
    'initialization': 'strict whole-model and whole-optimizer continuation; no module mixing or new learned parameters',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()},
    'sampling': 'every fourth batch real donor-uniform, otherwise synthetic subject-uniform and uniform eligible observation; frozen schedules',
    'loss': 'geometry-best mode five-point error/500um + .1 full mixtureNLL + .2 winnerCE; synthetic exact sparsePSF error/500um, masked clean appearance, .2 teacher field8, .1 fit feedback, .02 eligible predicted-row covarianceNLL; real weak pose .25 and fit feedback .02',
    'teacher': 'four random SYN rows per batch fit at canonical truth pose; other four use attached direct prediction; teacher covariance excluded',
    'optimizer': 'parent AdamW moments preserved, cosine lr .0001 -> .00002, weight decay .0001, global norm clip5, FP32',
    'augmentation': 'same spatial geometry; per-image exp(.10*N) gain, exp(.15*N) gamma, U[0,.02] additive noise; reapply known brush mask after noise so black exterior remains exact',
    'uncertainty': 'uncalibrated; joint covariance trained only synthetic predicted rows within .35rad and1500um, never teacher rows or weak real reference',
    'constraints': 'all optional context/marks absent; constraint support remains untrained and unqualified',
    'real_chart': 'canonical normal has positive largest-absolute component, matching synthetic sampler; if observed normal has negative dominant component, Ocanonical=Oobserved+(W-1)/W*Uobserved, Ucanonical=-Uobserved, Vcanonical=Vobserved and horizontal_reflection=1; observed pixels/affine remain unchanged',
    'scope': 'internal development only; one-atlas synthetic subjects and weak Allen affines are not independent expert validation',
    'source_sha256': {}, 'data_sha256': {},
}
for name in ('train_joint_v7_expanded', 'arbitrary_plane_joint_model_v7', 'arbitrary_plane_full_frame_primitives',
             'arbitrary_plane_geometry', 'arbitrary_plane_ribbon_v6', 'arbitrary_plane_recurrent_model',
             'arbitrary_plane_joint_uncertainty', 'arbitrary_plane_allen_atlas_binding_v6'):
    path = repository / 'training' / f'{name}.py'
    shutil.copyfile(path, RUN / path.name)
    config['source_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path, expected=None):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    config['data_sha256'][str(path)] = digest
    return digest


for directory in (PARENT, SYNTHETIC, REAL, OLD):
    bind(directory / 'completed.json')
bind(SYNTHETIC / 'protocol.json', synthetic_completed['protocol_sha256'])
syn = {key: [] for key in ('image', 'clean', 'support', 'retained', 'mask', 'outline', 'state', 'slab', 'field', 'offsets', 'weights', 'reflection', 'eligible')}
synthetic_ids, subject_choices, first_pixel_y, first_pixel_x = [], [], None, None
print('Loading only fully completed synthetic subject shards and exact sparse coordinate targets', flush=True)
for shard in synthetic_completed['shards']:
    directory = SYNTHETIC / shard['directory']
    bind(directory / 'completed.json', shard['completed_sha256'])
    bind(directory / 'subject.json', shard['subject_sha256'])
    bind(directory / 'sections.jsonl', shard['sections_sha256'])
    subject = json.loads((directory / 'subject.json').read_text())
    assert subject['lineage']['split'] == 'train' and shard['sections'] == 512
    choices = []
    for record in map(json.loads, (directory / 'sections.jsonl').read_text().splitlines()):
        path = directory / record['file']
        bind(path, record['sha256'])
        index = len(synthetic_ids)
        with np.load(path) as a:
            state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(a['canonical_ouv_um']))).float()
            if first_pixel_y is None:
                first_pixel_y, first_pixel_x = a['psf_pixel_y'].copy(), a['psf_pixel_x'].copy()
            assert np.array_equal(first_pixel_y, a['psf_pixel_y']) and np.array_equal(first_pixel_x, a['psf_pixel_x'])
            for key, array in (('image', 'shared_pre_brush_image'), ('clean', 'clean_atlas_intensity'),
                    ('support', 'finite_support'), ('retained', 'tissue_retained'), ('mask', 'brush_keep_mask'),
                    ('outline', 'brush_outline'), ('slab', 'target_psf_um'), ('field', 'canonical_field_8'),
                    ('offsets', 'offsets_um'), ('weights', 'weights'), ('reflection', 'horizontal_reflection'), ('eligible', 'eligible')):
                syn[key].append(torch.from_numpy(a[array].copy()))
            syn['state'].append(state)
            choices.extend(index * 3 + np.flatnonzero(a['eligible']))
        synthetic_ids.append({**subject['lineage'], 'section_id': record['section_id'], 'file': str(path), 'sha256': record['sha256']})
    subject_choices.append(torch.tensor(choices))
syn = {key: torch.stack(value) for key, value in syn.items()}
assert len(synthetic_ids) == 4096 and syn['image'].shape[1:] == (SIDE, SIDE)
pixel_y = torch.from_numpy(first_pixel_y).cuda()
pixel_x = torch.from_numpy(first_pixel_x).cuda()
real_ids = [json.loads(line) for line in (REAL / 'records.jsonl').read_text().splitlines()]
for name, expected in real_completed['output_sha256'].items():
    bind(REAL / name, expected)
real_images = torch.from_numpy(np.load(REAL / 'images.npy'))
assert real_images.shape == (1344, 1, SIDE, SIDE)
with np.load(REAL / 'geometry.npz') as arrays:
    real_state = torch.from_numpy(arrays['state'].copy())
    real_reflection = torch.from_numpy(arrays['reflection'].copy())
    real_thickness = torch.from_numpy(arrays['thickness_um'].copy())
    real_train = torch.from_numpy(arrays['is_train'].copy())
    observed_affine = torch.from_numpy(arrays['model_pixel_to_ap_dv_ml_um'].copy()).double()
real_normal = full_frame_state_to_components(real_state)[1][..., :, 2]
real_reflection = (real_normal.gather(1, real_normal.abs().argmax(-1)[:, None])[:, 0] < 0).long()
real_ouv = frame_to_physical_ouv(*full_frame_state_to_components(real_state)).reshape(-1, 3, 3).double()
reflected = real_reflection.bool()
real_ouv[reflected, 0] += (SIDE - 1) / SIDE * real_ouv[reflected, 1]
real_ouv[reflected, 1] *= -1
real_state = full_frame_state_from_components(*physical_ouv_to_frame(real_ouv)).float()
sample_pixels = torch.tensor([[0., 0.], [SIDE - 1., 0.], [0., SIDE - 1.], [SIDE - 1., SIDE - 1.], [(SIDE - 1.) / 2] * 2], dtype=torch.float64)
canonical_pixels = sample_pixels[None].repeat(len(real_state), 1, 1)
canonical_pixels[reflected, :, 0] = SIDE - 1 - canonical_pixels[reflected, :, 0]
reconstructed = real_ouv[:, None, 0] + torch.einsum('bpi,bij->bpj', canonical_pixels / SIDE, real_ouv[:, 1:])
observed = torch.einsum('pi,bji->bpj', torch.cat((sample_pixels, torch.ones(5, 1)), -1), observed_affine)
real_chart_error = float((reconstructed - observed).norm(dim=-1).max())
assert real_chart_error < .05, 'canonical reflection changed the observed physical affine'
config['real_chart_reflected_rows'] = reflected.nonzero()[:, 0].tolist()
config['real_chart_max_five_point_error_um'] = real_chart_error
real_donors = sorted({r['animal_id'] for r in real_ids if r['training_split'] == 'train'})
real_dev_donors = {r['animal_id'] for r in real_ids if r['training_split'] == 'development'}
assert len(real_donors) == 58 and len(real_dev_donors) == 6 and not set(real_donors) & real_dev_donors
assert int(real_train.sum()) == 1280
real_choices = [torch.tensor([i for i, r in enumerate(real_ids) if r['animal_id'] == donor]) for donor in real_donors]

old_images, old_state, old_eligible, old_ids = [], [], [], []
for record in json.loads((OLD / 'completed.json').read_text())['sections']:
    if record['lineage']['split'] != 'development':
        continue
    for name, expected in record['artifact_sha256'].items():
        bind(OLD / name, expected)
    meta = json.loads((OLD / record['artifacts']['metadata']).read_text())
    with np.load(OLD / record['artifacts']['arrays']) as a:
        ouv = a[meta['canonical_anatomy_plane_fit']['arrays']['physical_ouv_ap_dv_ml_um_float64']['__ndarray__']]
        state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(ouv))).float()
        for observation in meta['observations']:
            old_images.append(torch.from_numpy(a[observation['image_outline_availability_float32']['__ndarray__']].copy()))
            old_state.append(state)
            old_eligible.append(observation['support_information_eligible'])
            old_ids.append({**observation['lineage'], 'mode': observation['selected_mode']})
old_images = torch.cat((torch.stack(old_images), torch.zeros(len(old_images), 2, 96, 96)), 1)
old_state, old_eligible = torch.stack(old_state), torch.tensor(old_eligible)
assert not {r['animal_id'] for r in synthetic_ids} & {r['animal_id'] for r in old_ids}
assert len({r['animal_id'] for r in old_ids}) == 4

generator = torch.Generator().manual_seed(SEED)
is_real = torch.arange(1, STEPS + 1) % 4 == 0
schedule = torch.empty(STEPS, BATCH, dtype=torch.long)
for real_batch, choices in ((False, subject_choices), (True, real_choices)):
    updates = (is_real == real_batch).nonzero()[:, 0]
    group = torch.randint(len(choices), (len(updates), BATCH), generator=generator)
    selected = torch.empty_like(group)
    for index, choice in enumerate(choices):
        mask = group == index
        selected[mask] = choice[torch.randint(len(choice), (int(mask.sum()),), generator=generator)]
    schedule[updates] = selected
teacher = torch.rand(STEPS, BATCH, generator=generator).argsort(-1).argsort(-1) < BATCH // 2
teacher[is_real] = False
augmentation = torch.cat((torch.randn(STEPS, BATCH, 2, generator=generator) * torch.tensor([.10, .15]),
                          torch.rand(STEPS, BATCH, 1, generator=generator) * .02), -1)
noise_seeds = torch.randint(0, 2**31 - 1, (STEPS,), generator=generator)
torch.save({'is_real': is_real, 'observation_index': schedule, 'teacher': teacher,
            'augmentation_log_gain_log_gamma_noise_std': augmentation, 'noise_seed': noise_seeds}, RUN / 'schedule.pt')
(RUN / 'identities.json').write_text(json.dumps({'synthetic_sections': synthetic_ids, 'synthetic_mode_order': ['raw', 'exact_black', 'imperfect_brush'],
    'real_rows': real_ids, 'old_synthetic_development_observations': old_ids}, indent=2), encoding='utf8')
bind(PARENT / 'joint_step_06000.pt')
checkpoint = torch.load(PARENT / 'joint_step_06000.pt', map_location='cpu', weights_only=True)
for name, expected in checkpoint['config']['source_sha256'].items():
    if not name.startswith('train_joint_v7'):
        assert config['source_sha256'][name] == expected, name
model = JointSliceModel().cuda()
model.load_state_dict(checkpoint['model_state'], strict=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state'])
parent_step = int(checkpoint['step'])
config['parent_step'] = parent_step
config['parameters'] = sum(p.numel() for p in model.parameters())
config['schedule_sha256'] = hashlib.sha256((RUN / 'schedule.pt').read_bytes()).hexdigest()
config['identities_sha256'] = hashlib.sha256((RUN / 'identities.json').read_bytes()).hexdigest()
(RUN / 'experiment.json').write_text(json.dumps(config, indent=2), encoding='utf8')
del checkpoint
atlas_array, annotation = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array[:1]).cuda()
del atlas_array, annotation
torch.manual_seed(SEED + 1)
torch.cuda.manual_seed_all(SEED + 1)


def plane_points(state, side):
    centre, frame, basis = full_frame_state_to_components(state)
    end = (side - 1) / side
    uv = state.new_tensor([[0., 0.], [end, 0.], [0., end], [end, end], [end / 2, end / 2]]) - .5
    return centre[..., None, :] + torch.einsum('...ij,pj->...pi', frame[..., :2] @ basis, uv)


def synthetic_batch(observation):
    index, mode = observation // 3, observation % 3
    mask = syn['mask'][index, mode]
    image = syn['image'][index] * mask
    inputs = torch.stack((image, syn['outline'][index, mode].float(),
        (mode != 0)[:, None, None].expand(-1, SIDE, SIDE).float(), torch.zeros_like(image), torch.zeros_like(image)), 1)
    return inputs, index, mask


def save_checkpoint(stage_step):
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'step': parent_step + stage_step, 'stage_step': stage_step, 'config': config,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'calibrated': False}, RUN / f'joint_step_{parent_step + stage_step:05d}.pt')


def evaluate(stage_step):
    model.eval()
    fixed_train = torch.cat([choice[torch.linspace(0, len(choice) - 1, 8).long()] for choice in subject_choices])
    train_inputs, train_index, _ = synthetic_batch(fixed_train)
    real_dev = (~real_train).nonzero()[:, 0]
    real_input = torch.cat((real_images[real_dev].float(), torch.zeros(len(real_dev), 4, SIDE, SIDE)), 1)
    datasets = (('synthetic_train_fixed', train_inputs, syn['state'][train_index], torch.ones(len(train_index), dtype=torch.bool),
                 [synthetic_ids[i]['animal_id'] for i in train_index]),
                ('synthetic_development', old_images, old_state, old_eligible, [r['animal_id'] for r in old_ids]),
                ('real_development_weak', real_input, real_state[real_dev], torch.ones(len(real_dev), dtype=torch.bool),
                 [real_ids[i]['animal_id'] for i in real_dev]))
    summaries = {}
    with torch.no_grad():
        for name, inputs, truth, eligible, animals in datasets:
            outputs = {key: [] for key in ('state', 'log_mass', 'std', 'concentration', 'reflection_logit')}
            for batch in inputs.split(8):
                prediction = model.predict(batch.cuda())
                for key in outputs:
                    outputs[key].append(prediction[key].cpu())
            outputs = {key: torch.cat(value) for key, value in outputs.items()}
            state = outputs['state']
            mode = outputs['log_mass'].argmax(-1)
            coordinate = (plane_points(state, inputs.shape[-1]) - plane_points(truth, inputs.shape[-1])[:, None]).norm(dim=-1).mean(-1)
            oracle = coordinate.argmin(-1)
            normal = full_frame_state_to_components(state)[1][..., :, 2]
            target_normal = full_frame_state_to_components(truth)[1][..., :, 2]
            normal_error = torch.rad2deg((normal * target_normal[:, None]).sum(-1).abs().clamp(0, 1).acos())
            centre_error = (state[..., :3] - truth[:, None, :3]).norm(dim=-1)
            rows = torch.arange(len(state))
            np.savez(RUN / f'{name}_step_{stage_step:05d}.npz', **{key: value.numpy() for key, value in outputs.items()},
                target_state=truth.numpy(), eligible=eligible.numpy(), map_mode=mode.numpy(), geometry_best_mode=oracle.numpy(),
                coordinate_error_um=coordinate.numpy(), normal_error_deg=normal_error.numpy(), centre_error_um=centre_error.numpy(),
                animal_id=np.asarray(animals), observation_index=fixed_train.numpy() if name == 'synthetic_train_fixed' else
                (real_dev.numpy() if name == 'real_development_weak' else np.arange(len(state))))
            summary = {}
            for animal in sorted(set(animals)):
                keep = torch.tensor([value == animal for value in animals]) & eligible
                summary[str(animal)] = {'observations': int(keep.sum()),
                    'map_normal_deg': float(normal_error[rows, mode][keep].mean()),
                    'map_centre_um': float(centre_error[rows, mode][keep].mean()),
                    'map_coordinate_um': float(coordinate[rows, mode][keep].mean()),
                    'oracle_coordinate_um': float(coordinate[rows, oracle][keep].mean())}
            summaries[name] = summary
    (RUN / f'development_summary_{stage_step:05d}.json').write_text(json.dumps(summaries, indent=2), encoding='utf8')
    compact = {name: {key: float(np.mean([row[key] for row in animals.values()]))
                     for key in ('map_normal_deg', 'map_centre_um', 'map_coordinate_um', 'oracle_coordinate_um')}
               for name, animals in summaries.items()}
    print(json.dumps({'event': 'animal_equal_direct_readout', 'stage_step': stage_step, **compact}), flush=True)
    model.train()


save_checkpoint(0)
evaluate(0)
started = time.perf_counter()
with (RUN / 'training.jsonl').open('w', encoding='utf8') as trace:
    for step in range(1, STEPS + 1):
        rows = schedule[step - 1]
        real_batch = bool(is_real[step - 1])
        if real_batch:
            inputs = torch.cat((real_images[rows].float(), torch.zeros(BATCH, 4, SIDE, SIDE)), 1)
            masks = torch.ones(BATCH, SIDE, SIDE, dtype=torch.bool)
            truth, reflection = real_state[rows].cuda(), real_reflection[rows].cuda()
            offsets = real_thickness[rows, None].cuda() * torch.linspace(-.5, .5, 9, device='cuda')
            weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda').float()[None].expand(BATCH, -1) / 16
        else:
            inputs, index, masks = synthetic_batch(rows)
            truth, reflection = syn['state'][index].cuda(), syn['reflection'][index].cuda().long()
            offsets, weights = syn['offsets'][index].cuda(), syn['weights'][index].cuda()
        aug = augmentation[step - 1]
        noise = torch.randn(BATCH, SIDE, SIDE, generator=torch.Generator().manual_seed(int(noise_seeds[step - 1])))
        inputs[:, 0] = ((inputs[:, 0].clamp_min(0).pow(aug[:, 1, None, None].exp()) * aug[:, 0, None, None].exp()
                         + noise * aug[:, 2, None, None]).clamp(0, 1) * masks)
        inputs = inputs.cuda()
        for group in optimizer.param_groups:
            group['lr'] = 2e-5 + 8e-5 * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        optimizer.zero_grad(set_to_none=True)
        prediction = model.predict(inputs)
        distance = (plane_points(prediction['state'], SIDE) - plane_points(truth, SIDE)[:, None]).norm(dim=-1).mean(-1)
        best = distance.detach().argmin(-1)
        batch_rows = torch.arange(BATCH, device='cuda')
        physical_loss = distance[batch_rows, best].mean() / 500
        pose_nll = -model.component_log_prob(prediction, truth, reflection).logsumexp(-1).mean()
        mode_loss = F.nll_loss(prediction['log_mass'], best)
        pose_loss = physical_loss + .1 * pose_nll + .2 * mode_loss
        teachers = teacher[step - 1].cuda()
        initial = torch.where(teachers[:, None], truth, prediction['state'][batch_rows, best])[:, None]
        fitted = model.fit(prediction, atlas, offsets, weights, best[:, None], reflection[:, None], initial_state=initial)
        feedback_per_row = fitted['mismatch'][:, 0] + .05 * fitted['difficulty'][:, 0]
        feedback = feedback_per_row[~teachers].mean()
        appearance_loss = dense_loss = field_loss = uncertainty_loss = pose_loss.new_zeros(())
        if real_batch:
            loss = .25 * pose_loss + .02 * feedback
        else:
            q = (syn['support'][index] * syn['retained'][index] * masks).cuda()
            valid = (1 - syn['support'][index].cuda() + q).clamp(0, 1)
            clean = syn['clean'][index].cuda()
            difference = prediction['appearance'][:, 0] - clean
            appearance_loss = ((difference.abs() + .5 * difference.square()) * valid).sum() / valid.sum().clamp_min(1)
            predicted_slab = fitted['coordinates'][:, :, pixel_y[:, None], pixel_x[None, :], :]
            sparse_q = q[:, pixel_y[:, None], pixel_x[None, :]]
            error = (predicted_slab - syn['slab'][index].cuda()) / 500
            dense_per_row = ((error.square().sum(-1) + 1e-4).sqrt() * sparse_q[:, None] * weights[:, :, None, None]).sum((1, 2, 3)) / sparse_q.sum((1, 2)).clamp_min(1)
            dense_loss = dense_per_row.mean()
            geometry = fitted['geometry']
            field = torch.cat((geometry['residual_local_um'] / 200, geometry['director_delta_local'] / .1), 1)
            field = F.interpolate(field, (8, 8), mode='bilinear', align_corners=False)
            target_field = syn['field'][index].cuda()
            field_loss = F.smooth_l1_loss(field[teachers], target_field[teachers])
            loss = pose_loss + appearance_loss + dense_loss + .2 * field_loss + .1 * feedback
            centre, rotation, basis = full_frame_state_to_components(fitted['state'].detach())
            target_centre, target_rotation, target_basis = full_frame_state_to_components(truth)
            relative = rotation.transpose(-2, -1) @ target_rotation
            angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
            local = ((~teachers) & (angle < .35) & ((target_centre - centre).norm(dim=-1) < 1500)).nonzero()[:, 0]
            if len(local):
                b = torch.linalg.solve(basis[local], target_basis[local])
                delta = torch.cat((local_rotation_log(relative[local]) / .1,
                    torch.einsum('bji,bj->bi', rotation[local], target_centre[local] - centre[local]) / 500,
                    b.diagonal(dim1=-2, dim2=-1).log() / .1, (b[:, 0, 1] / b[:, 1, 1])[:, None] / .1), -1)
                residual = torch.cat((delta, (target_field[local] - field[local]).flatten(1)), -1).detach()
                distribution = torch.distributions.LowRankMultivariateNormal(torch.zeros_like(residual),
                    fitted['joint_factor'][local], fitted['joint_std'][local].square())
                uncertainty_loss = -distribution.log_prob(residual).mean() / residual.shape[-1]
                loss = loss + .02 * uncertainty_loss
            if step == 1:
                gradient = torch.autograd.grad(feedback, model.pose[-1].weight, retain_graph=True)[0]
                feedback_gradient = float(gradient.norm())
                assert np.isfinite(feedback_gradient) and feedback_gradient > 0
                print(json.dumps({'event': 'predicted_pose_receives_fitting_feedback', 'gradient_norm': feedback_gradient}), flush=True)
        assert torch.isfinite(loss), f'nonfinite loss at stage step{step}'
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'stage_step': step, 'total_step': parent_step + step, 'real_weak_batch': real_batch,
            'observation_indices': rows.tolist(), 'teacher_rows': teacher[step - 1].tolist(),
            'loss': float(loss.detach()), 'physical_um': float(physical_loss.detach() * 500),
            'pose_nll': float(pose_nll.detach()), 'mode_ce': float(mode_loss.detach()),
            'appearance': float(appearance_loss.detach()), 'sparse_psf_um': float(dense_loss.detach() * 500),
            'teacher_field': float(field_loss.detach()), 'feedback': float(feedback.detach()),
            'joint_uncertainty_nll': float(uncertainty_loss.detach()), 'gradient_norm': float(gradient),
            'learning_rate': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter() - started}
        trace.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            trace.flush()
            print(json.dumps(row), flush=True)
        if step % 4000 == 0:
            save_checkpoint(step)
            evaluate(step)
for name, expected in config['source_sha256'].items():
    assert hashlib.sha256((repository / 'training' / name).read_bytes()).hexdigest() == expected
(RUN / 'completed.json').write_text(json.dumps({'stage_updates': STEPS, 'total_updates': parent_step + STEPS,
    'seconds': time.perf_counter() - started, 'calibrated': False, 'scope': config['scope']}, indent=2), encoding='utf8')
print('Expanded joint training complete; frozen results are ready for inspection', flush=True)
