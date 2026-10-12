"""Matched TRAIN-inner candidate-aligned anatomy ranker against a support null."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.oriented_slab_ranker_299 import make_ranker_299, render_bank_299, score_bank_299


SEED, SIDE, UPDATES = 20261012299, 192, 1200
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/ORIENTED_SLAB_RANKER_299_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt'
RUN = ROOT / 'runs/oriented_slab_ranker_299_train_inner_001'
EXPECTED_PARENT = 'b4689e1d0dcbf1e8d65a63a400962bf27f44b2e571c8361e29e0e51412b2b97f'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def draw(context, base, seed, variant):
    subject = base * VARIANTS + variant
    with torch.no_grad():
        original = sample_one_shot_slide_artifacts_v9(context, [subject], seed, SIDE)
        if not bool(original['eligible'][0]):
            return original
        return sample_postartifact_gauge_v1(context, [subject], seed, SIDE,
                                            source_sample=original)


def candidates(proposer, sample):
    with torch.no_grad():
        prediction = proposer.predict(sample['inputs'])
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
        order = prior.argsort(descending=True)[:16]
        states = prediction['state'][0, order // 2]
        reflections = order % 2
        pixels = sample['valid_mask'][0].flatten().nonzero().flatten()
        pixels = pixels[torch.linspace(0, len(pixels) - 1, min(256, len(pixels)),
            device=pixels.device).long()]
        chart = torch.stack((pixels.remainder(SIDE), pixels.div(SIDE,
            rounding_mode='floor')), -1).float() / SIDE
        truth = sample['centre'][0].reshape(-1, 3)[pixels]
        points = rigid_points_090(states[None], reflections[None], chart[None],
                                  (SIDE, SIDE))[0]
        error = (points - truth[None]).norm(dim=-1).mean(-1) / 1000
    return states, reflections, prior[order], order, error


def matched_pair(error, support, prior):
    near = int(error.argmin())
    allowed = ((error >= max(2., float(error[near]) + 1.))
               & ((support - support[near]).abs() <= .10))
    if error[near] > 1.5 or not bool(allowed.any()):
        return None
    far = int(torch.where(allowed, prior, -torch.inf).argmax())
    return near, far


assert sha(PARENT) == EXPECTED_PARENT
context = load_streaming_synthetic_v7_64('cuda')
saved = torch.load(PARENT, map_location='cpu', weights_only=False)
assert saved['step'] == 8000 and saved['calibrated'] is False
proposer = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval().requires_grad_(False)
proposer.load_state_dict(saved['model'], strict=True)
del saved

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
full = make_ranker_299().cuda()
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
support_only = make_ranker_299().cuda()
assert all(torch.equal(x, y) for x, y in zip(full.parameters(), support_only.parameters()))
rankers = {'full': full, 'support_only': support_only}
optimizers = {name: torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
              for name, model in rankers.items()}
train_rng = np.random.default_rng(SEED + 1)
panel_rng = np.random.default_rng(SEED + 2)
config = {'candidate': 299, 'seed': SEED, 'updates': UPDATES, 'side': SIDE,
    'train_bases': [0, 55], 'inner_bases': [56, 63], 'inner_sections_per_base': 8,
    'parent_checkpoint_sha256': EXPECTED_PARENT,
    'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'train_oriented_slab_ranker_299.py', 'oriented_slab_ranker_299.py',
        'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_postartifact_gauge_v1.py',
        'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_full_frame_primitives.py')},
    'synthetic_provenance': context['provenance'],
    'random_initialized_ranker': True, 'frozen_proposer': True,
    'external_weights_features_or_pseudolabels': False,
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False,
    'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

started = time.perf_counter()
panel, panel_ids, panel_attempts = [], set(), 0
with (RUN / 'inner_draws.jsonl').open('w', encoding='utf8') as output:
    for base in range(56, 64):
        for slot in range(8):
            while True:
                panel_attempts += 1
                seed = SEED + 1_000_000_000 + panel_attempts
                variant = int(panel_rng.integers(VARIANTS))
                sample = draw(context, base, seed, variant)
                identity = sample['provenance'][0]
                used = bool(sample['eligible'][0]) and identity['physical_section_id'] not in panel_ids
                output.write(json.dumps({'base': base, 'slot': slot, 'seed': seed,
                    'variant': variant, 'used': used, 'provenance': identity}, allow_nan=False) + '\n')
                if used:
                    break
                del sample
            panel_ids.add(identity['physical_section_id'])
            states, reflections, prior, order, error = candidates(proposer, sample)
            bank = render_bank_299(context['atlas'], states, reflections,
                                   sample['offsets'].expand(16, -1),
                                   sample['weights'].expand(16, -1))
            central_support = bank[:, 2, 1].mean((1, 2))
            pair = matched_pair(error, central_support, prior)
            normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
            panel.append({'physical_section_id': identity['physical_section_id'],
                'base': base, 'slot': slot, 'seed': seed, 'variant': variant,
                'provenance': identity, 'inputs': sample['inputs'][0].cpu(),
                'states': states.cpu(), 'reflections': reflections.cpu(),
                'branch_ids': order.cpu(), 'prior': prior.cpu(), 'error_mm': error.cpu(),
                'central_support': central_support.cpu(), 'bank': bank.cpu(),
                'pair': pair, 'brush_available': bool(sample['inputs'][0, 2, 0, 0]),
                'obliquity_deg': float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1))))})
            del sample, bank
assert len(panel) == len(panel_ids) == 64
torch.save(panel, RUN / 'inner_panel.pt')
print(json.dumps({'event': 'panel_frozen', 'cases': len(panel),
                  'matched_pairs': sum(item['pair'] is not None for item in panel)}), flush=True)

accepted, attempted = 0, 0
with (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws, \
        (RUN / 'training.jsonl').open('w', encoding='utf8') as training:
    while accepted < UPDATES:
        assert time.perf_counter() - started < 7200, 'predeclared two-hour hard stop'
        attempted += 1
        base = int(train_rng.integers(56))
        variant = int(train_rng.integers(VARIANTS))
        seed = SEED + 2_000_000_000 + attempted
        sample = draw(context, base, seed, variant)
        identity = sample['provenance'][0]
        eligible = bool(sample['eligible'][0])
        pair, bank = None, None
        if eligible:
            states, reflections, prior, order, error = candidates(proposer, sample)
            centre_bank = render_bank_299(context['atlas'], states, reflections,
                sample['offsets'].expand(16, -1), sample['weights'].expand(16, -1), (0.,))
            central_support = centre_bank[:, 0, 1].mean((1, 2))
            pair = matched_pair(error, central_support, prior)
            if pair is not None:
                near, far = pair
                extras = [int(index) for index in range(16) if index not in pair]
                random_far = extras[int(train_rng.integers(len(extras)))]
                chosen = list(dict.fromkeys((near, far, 0, random_far)))
                bank = render_bank_299(context['atlas'], states[chosen], reflections[chosen],
                    sample['offsets'].expand(len(chosen), -1),
                    sample['weights'].expand(len(chosen), -1))
        draws.write(json.dumps({'attempt': attempted, 'accepted': pair is not None,
            'base': base, 'variant': variant, 'seed': seed,
            'physical_section_id': identity['physical_section_id'],
            'provenance': identity, 'near_far': pair,
            'candidate_error_mm': error.tolist() if eligible else None,
            'candidate_support': central_support.tolist() if eligible else None},
            allow_nan=False) + '\n')
        if pair is None:
            del sample
            continue
        accepted += 1
        active = [(chosen.index(index), index) for index in chosen if
                  index != near and float(error[index] - error[near]) >= .75 and
                  abs(float(central_support[index] - central_support[near])) <= .12]
        target = F.avg_pool2d((sample['valid_mask'] &
            ~sample['fragment_displaced_mask']).float()[:, None], 4)
        losses = {}
        for name, model in rankers.items():
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            evidence, reliability = score_bank_299(model, sample['inputs'], bank,
                                                    name == 'support_only')
            near_pos = chosen.index(near)
            pair_loss = torch.stack([
                F.softplus(4 * (.15 - (evidence[near_pos] - evidence[position])))
                + F.softplus(2 * (.25 - (prior[near] - prior[index]
                    + 4 * (evidence[near_pos] - evidence[position]))))
                for position, index in active]).mean()
            tissue_loss = F.binary_cross_entropy_with_logits(reliability,
                (target > .7).float())
            loss = pair_loss + .10 * tissue_loss
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            losses[name] = {'pair': float(pair_loss.detach()),
                'reliability': float(tissue_loss.detach()),
                'gradient_norm': float(gradient.detach()),
                'near_evidence': float(evidence[near_pos].detach()),
                'far_evidence': float(evidence[chosen.index(far)].detach())}
        training.write(json.dumps({'step': accepted, 'attempt': attempted,
            'physical_section_id': identity['physical_section_id'],
            'base': base, 'variant': variant, 'seed': seed,
            'near_far': pair, 'chosen': chosen, 'errors_mm': error[chosen].tolist(),
            'support_fraction': central_support[chosen].tolist(),
            'losses': losses, 'elapsed_seconds': time.perf_counter() - started},
            allow_nan=False) + '\n')
        if accepted == 1 or accepted % 200 == 0:
            draws.flush(); training.flush()
            print(json.dumps({'event': 'train_milestone', 'accepted': accepted,
                'attempted': attempted, 'elapsed_seconds': time.perf_counter() - started,
                'losses': losses}), flush=True)
        del sample, bank

for model in rankers.values():
    model.eval()
rows = []
with (RUN / 'readout.jsonl').open('w', encoding='utf8') as output:
    with torch.inference_mode():
        for item in panel:
            inputs, bank = item['inputs'][None].cuda(), item['bank'].cuda()
            prior, error = item['prior'].cuda(), item['error_mm'].cuda()
            near_far = item['pair']
            arms = {}
            for name, model in rankers.items():
                evidence, _ = score_bank_299(model, inputs, bank, name == 'support_only')
                selected = int((prior + 4 * evidence).argmax())
                arms[name] = {'evidence': evidence.tolist(), 'selected': selected,
                    'selected_error_mm': float(error[selected]),
                    'near_far_evidence_win': bool(evidence[near_far[0]] > evidence[near_far[1]])
                        if near_far is not None else None}
            row = {'physical_section_id': item['physical_section_id'],
                'base': item['base'], 'slot': item['slot'],
                'brush_available': item['brush_available'],
                'obliquity_deg': item['obliquity_deg'],
                'prior_error_mm': float(error[0]),
                'oracle_error_mm': float(error.min()),
                'candidate_error_mm': error.tolist(), 'candidate_prior_log': prior.tolist(),
                'candidate_support': item['central_support'].tolist(),
                'near_far': near_far, 'arms': arms}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            if len(rows) % 16 == 0:
                print(json.dumps({'event': 'readout_milestone', 'cases': len(rows)}), flush=True)

eligible = [row for row in rows if row['near_far'] is not None]
summary = {'n': len(rows), 'matched_pairs': len(eligible),
    'prior_mean_mm': float(np.mean([row['prior_error_mm'] for row in rows])),
    'oracle_mean_mm': float(np.mean([row['oracle_error_mm'] for row in rows]))}
for name in rankers:
    summary[name] = {'mean_mm': float(np.mean([row['arms'][name]['selected_error_mm']
        for row in rows])), 'near_far_wins': sum(row['arms'][name]['near_far_evidence_win']
        for row in eligible), 'per_base_mean_mm': {str(base): float(np.mean([
        row['arms'][name]['selected_error_mm'] for row in rows if row['base'] == base]))
        for base in range(56, 64)}}
base_prior = {str(base): float(np.mean([row['prior_error_mm'] for row in rows
                                     if row['base'] == base])) for base in range(56, 64)}
gate = {'at_least_30_matched_pairs': len(eligible) >= 30,
    'full_evidence_near_far_at_least_75_percent':
        summary['full']['near_far_wins'] >= .75 * len(eligible),
    'full_evidence_near_far_gain_at_least_10_points':
        summary['full']['near_far_wins'] - summary['support_only']['near_far_wins'] >= .10 * len(eligible),
    'full_selected_gain_over_prior_at_least_0p20mm':
        summary['prior_mean_mm'] - summary['full']['mean_mm'] >= .20,
    'full_selected_gain_over_support_at_least_0p15mm':
        summary['support_only']['mean_mm'] - summary['full']['mean_mm'] >= .15,
    'at_least_six_of_eight_bases_nonworse_than_prior': sum(
        summary['full']['per_base_mean_mm'][str(base)] <= base_prior[str(base)]
        for base in range(56, 64)) >= 6}
summary['prior_per_base_mean_mm'] = base_prior
summary['gate'] = gate
summary['gate_pass'] = all(gate.values())
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
for name, model in rankers.items():
    torch.save({'model': model.state_dict(), 'step': UPDATES, 'arm': name,
        'parent_sha256': EXPECTED_PARENT, 'calibrated': False}, RUN / f'{name}.pt')
receipt = {'candidate': 299, 'attempted': attempted, 'accepted': accepted,
    'panel_attempted': panel_attempts, 'panel_count': len(panel), 'summary': summary,
    'checkpoint_sha256': {name: sha(RUN / f'{name}.pt') for name in rankers},
    'output_sha256': {name: sha(RUN / name) for name in (
        'config.json', 'inner_draws.jsonl', 'inner_panel.pt', 'draws.jsonl',
        'training.jsonl', 'readout.jsonl', 'summary.json')},
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False,
    'elapsed_seconds': time.perf_counter() - started}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
