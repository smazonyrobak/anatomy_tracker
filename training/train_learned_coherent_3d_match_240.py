"""TRAIN-inner anatomy descriptor pilot; the 226 proposer is frozen."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.local_3d_descriptor_240 import (
    Local3DDescriptor240, coherent_score_240, query_chart_240,
    render_candidate_bank_240, sample_candidate_features_240,
)
from training.support_aware_framing_217 import sample_support_aware_framing_217

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

source = Path(__file__).resolve().parent
run = root / 'runs/learned_coherent_3d_match_240_train_001'
protocol = source.parent / 'docs/publication/LEARNED_COHERENT_3D_MATCH_240_PROTOCOL_20261012.md'
parent = root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(parent) == 'dd76016764b1929a883f800f435e45e81fc503d3ee6bb76d66674e7c8e3e3fbc'
context = load_streaming_synthetic_v7_64(device='cuda')
saved = torch.load(parent, map_location='cpu', weights_only=True)
proposer = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(proposer, enabled=True)
proposer.load_state_dict(saved['model'], strict=True)
del saved

torch.manual_seed(20261012240)
torch.cuda.manual_seed_all(20261012240)
descriptor = Local3DDescriptor240().cuda()
optimizer = torch.optim.AdamW(descriptor.parameters(), lr=1e-4, weight_decay=1e-4)
train_rng = np.random.default_rng(20261012240)
validation_rng = np.random.default_rng(20261012241)
updates, interval, validation_count = 1024, 256, 32
best_loss, best_step, stale = float('inf'), 0, 0
draw_attempts = 0


def draw_case(base, appearance, seed):
    framed = sample_support_aware_framing_217(
        sample_panel_geometry_train_166(context, [base], seed, side=256), seed)
    sampler = sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else sample_one_shot_slide_artifacts_v7
    return sampler(context, [base], seed, side=256, source_sample=framed)


def prepare_case(sample, seed):
    with torch.no_grad():
        prediction = proposer.predict(sample['inputs'])
        states = prediction['state'][0].repeat_interleave(2, 0)
        reflections = torch.arange(2, device='cuda').repeat(proposer.modes)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        beam = prior.topk(160).indices
        pixels = sample['valid_mask'][0].flatten().nonzero().flatten()
        pixels = pixels[torch.linspace(0, len(pixels) - 1,
            min(128, len(pixels)), device='cuda').long()]
        sites = torch.stack((pixels.remainder(256),
            pixels.div(256, rounding_mode='floor')), -1).float() / 256
        tissue_truth = sample['centre'][0].reshape(-1, 3)[pixels]
        error = (rigid_points_090(states, reflections, sites) - tissue_truth[None]).norm(dim=-1).mean(-1) / 1000
        near = int(beam[error[beam].argmin()])
        wrong_choices = beam[:16][error[beam[:16]] >= error[near] + 1.]
        wrong = int(wrong_choices[0] if len(wrong_choices) else beam[:16][error[beam[:16]].argmax()])
        random = np.random.default_rng(np.random.SeedSequence([int(seed), 240]))
        perturbation = torch.tensor(random.uniform(-1, 1, 9), device='cuda', dtype=torch.float32)
        perturbation *= sample['state'].new_tensor((.12, .12, .12, 1300., 1300., 1300., .045, .045, .03))
        jittered = compose_full_frame_state(sample['state'], perturbation[None])
        query = query_chart_240(sample['inputs'].device, sample['inputs'].dtype)
        yx = ((query[:, [1, 0]] * 256).round().long()).clamp(0, 255)
        indices = yx[:, 0] * 256 + yx[:, 1]
        target_world = sample['centre'][0].reshape(-1, 3)[indices]
        target_valid = sample['valid_mask'][0].flatten()[indices]
        roles = [('jitter', jittered, sample['reflection'])]
        if bool((error[near] < 4) & (reflections[near] == sample['reflection'][0])):
            roles.append(('near', states[near:near + 1], reflections[near:near + 1]))
        if wrong != near:
            roles.append(('wrong', states[wrong:wrong + 1], reflections[wrong:wrong + 1]))
    return roles, target_world[None], target_valid[None], {
        'near_branch': int(near), 'wrong_branch': int(wrong),
        'near_mm': float(error[near]), 'wrong_mm': float(error[wrong]),
        'near_reflection_correct': bool(reflections[near] == sample['reflection'][0]),
        'valid_query_sites': int(target_valid.sum()),
    }


def case_loss(sample, seed):
    roles, target_world, target_valid, details = prepare_case(sample, seed)
    source_grid = descriptor.encode_source(sample['inputs'][:, :1])
    source_sites = source_grid[:, :, ::2, ::2].flatten(2).transpose(1, 2)
    loss_local, loss_coherent, scores, counts = [], [], {}, {}
    generator = torch.Generator(device='cuda').manual_seed(int(seed) % (2**63))
    displacement = F.normalize(torch.randn((1, 256, 12, 3), device='cuda',
        generator=generator), dim=-1) * torch.empty((1, 256, 12, 1),
        device='cuda').uniform_(1000, 3000, generator=generator)
    negative_world = (target_world[:, :, None] + displacement).flatten(1, 2)
    for role, state, reflection in roles:
        with torch.no_grad():
            bank, support = render_candidate_bank_240(context['atlas'], state,
                reflection, sample['offsets'], sample['weights'])
        atlas_grid = descriptor.encode_atlas(bank)
        initial, count = coherent_score_240(source_grid, atlas_grid, support,
            state, reflection, state)
        scores[role], counts[role] = initial, count
        if role != 'wrong':
            positive, positive_support = sample_candidate_features_240(
                atlas_grid, support, state, reflection, target_world)
            negatives, negative_support = sample_candidate_features_240(
                atlas_grid, support, state, reflection, negative_world)
            negatives = negatives.reshape(1, 256, 12, 64)
            negative_support = negative_support.reshape(1, 256, 12)
            keys = torch.cat((positive[:, :, None], negatives), 2)
            logits = 20 * (source_sites[:, :, None] * keys).sum(-1)
            logits = torch.cat((logits[:, :, :1],
                logits[:, :, 1:].masked_fill(negative_support <= .5, -1e4)), -1)
            usable = target_valid & (positive_support > .5) & (negative_support > .5).any(-1)
            if bool(usable.any()):
                loss_local.append(F.cross_entropy(logits[usable],
                    torch.zeros(int(usable.sum()), device='cuda', dtype=torch.long)))
            true_score, true_count = coherent_score_240(source_grid, atlas_grid, support,
                state, reflection, sample['state'])
            if true_count >= 32 and count >= 32:
                loss_coherent.append(F.softplus(5 * (initial - true_score) + .2))
        del atlas_grid
    if ('near' in scores and 'wrong' in scores
            and details['wrong_mm'] >= details['near_mm'] + .5
            and min(counts['near'], counts['wrong']) >= 32
            and abs(counts['near'] - counts['wrong']) <= 32):
        loss_coherent.append(F.softplus(5 * (scores['wrong'] - scores['near']) + .2))
    local = torch.stack(loss_local).mean() if loss_local else source_grid.sum() * 0
    coherent = torch.stack(loss_coherent).mean() if loss_coherent else source_grid.sum() * 0
    return local + .5 * coherent, {
        **details, 'local_ce': float(local.detach()), 'coherent_rank': float(coherent.detach()),
        'loss': float((local + .5 * coherent).detach()),
        'roles': [role for role, _, _ in roles],
        'score': {role: float(score.detach()) for role, score in scores.items()},
    }


validation = []
for index in range(validation_count):
    base = 48 + index % 4
    appearance = 'v4' if index % 2 == 0 else 'v7'
    while True:
        seed = int(validation_rng.integers(0, 2**63 - 1, dtype=np.int64))
        sample = draw_case(base, appearance, seed)
        if bool(sample['eligible'][0]):
            validation.append({'base': base, 'appearance': appearance, 'seed': seed,
                'physical_section_id': sample['provenance'][0]['physical_section_id'],
                'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest()})
            break

run.mkdir(parents=True, exist_ok=False)
(run / 'validation_manifest.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
config = {'seed': 20261012240, 'updates': updates, 'interval': interval,
    'validation_count': validation_count, 'train_bases': [0, 47], 'validation_bases': [48, 51],
    'sealed_readout_bases': [52, 55], 'unique_physical_section_per_presentation': True,
    'external_pretrained_weights_used': False, 'reserved_animal_used': False,
    'public_benchmark_used': False, 'calibrated': False, 'parent_sha256': sha(parent),
    'protocol_sha256': sha(protocol), 'source_sha256': {
        name: sha(source / name) for name in (
            'train_learned_coherent_3d_match_240.py', 'local_3d_descriptor_240.py',
            'arbitrary_plane_panel_geometry_train_166.py',
            'support_aware_framing_217.py', 'arbitrary_plane_one_shot_slide_artifacts_v4.py',
            'arbitrary_plane_one_shot_slide_artifacts_v7.py',
            'arbitrary_plane_one_shot_model.py', 'global_atlas_contrast_090.py')},
    'synthetic_provenance': context['provenance']}
(run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

started = time.perf_counter()
with (run / 'draws.jsonl').open('w', encoding='utf-8') as draws, \
        (run / 'training.jsonl').open('w', encoding='utf-8') as training, \
        (run / 'validation.jsonl').open('w', encoding='utf-8') as validations:
    for step in range(1, updates + 1):
        base = int(train_rng.integers(0, 48))
        appearance = 'v4' if step % 2 else 'v7'
        while True:
            seed = int(train_rng.integers(0, 2**63 - 1, dtype=np.int64))
            sample = draw_case(base, appearance, seed)
            draw_attempts += 1
            used = bool(sample['eligible'][0])
            draws.write(json.dumps({'step': step, 'used': used,
                **sample['provenance'][0]}, allow_nan=False) + '\n')
            if used:
                break
        descriptor.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = case_loss(sample, seed)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(descriptor.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        training.write(json.dumps({'step': step, 'base': base, 'appearance': appearance,
            'seed': seed, 'physical_section_id': sample['provenance'][0]['physical_section_id'],
            **metrics, 'gradient_norm': float(norm), 'elapsed_seconds': time.perf_counter() - started},
            allow_nan=False) + '\n')
        if step % interval:
            continue
        descriptor.eval()
        rows = []
        with torch.no_grad():
            for item in validation:
                case = draw_case(item['base'], item['appearance'], item['seed'])
                assert case['provenance'][0]['physical_section_id'] == item['physical_section_id']
                assert hashlib.sha256(case['inputs'].cpu().numpy().tobytes()).hexdigest() == item['input_sha256']
                _, row = case_loss(case, item['seed'])
                rows.append({'physical_section_id': item['physical_section_id'],
                    'appearance': item['appearance'], **row})
        mean_loss = float(np.mean([r['loss'] for r in rows]))
        improved = mean_loss < best_loss - .002
        if improved:
            best_loss, best_step, stale = mean_loss, step, 0
            torch.save({'descriptor': descriptor.state_dict(), 'step': step,
                'parent_sha256': config['parent_sha256'], 'config': config}, run / 'best.pt')
        else:
            stale += 1
        validations.write(json.dumps({'step': step, 'mean_loss': mean_loss,
            'v4_mean_loss': float(np.mean([r['loss'] for r in rows if r['appearance'] == 'v4'])),
            'v7_mean_loss': float(np.mean([r['loss'] for r in rows if r['appearance'] == 'v7'])),
            'best_step': best_step, 'improved': improved, 'rows': rows}, allow_nan=False) + '\n')
        draws.flush(); training.flush(); validations.flush()
        print(json.dumps({'event': 'validation', 'step': step, 'mean_loss': mean_loss,
            'best_step': best_step, 'stale': stale, 'elapsed_seconds': time.perf_counter() - started}),
            flush=True)
        if stale >= 2:
            break

(run / 'completed.json').write_text(json.dumps({'updates': step,
    'draw_attempts': draw_attempts, 'best_step': best_step,
    'best_validation_loss': best_loss, 'training_exited': True,
    'external_pretrained_weights_used': False, 'reserved_animal_used': False,
    'public_benchmark_used': False, 'calibrated': False,
    'checkpoint_sha256': sha(run / 'best.pt'),
    'output_sha256': {name: sha(run / name) for name in (
        'config.json', 'validation_manifest.json', 'draws.jsonl',
        'training.jsonl', 'validation.jsonl')}}, indent=2), encoding='utf-8')
print(json.dumps({'event': 'completed', 'updates': step,
    'best_step': best_step, 'best_validation_loss': best_loss}), flush=True)
