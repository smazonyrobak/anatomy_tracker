"""Matched exact-point-supervised follow-up to failed oriented-slab ranker 299."""

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

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.oriented_slab_correspondence_300 import point_loss_300
from training.oriented_slab_ranker_299 import make_ranker_299, render_bank_299, score_bank_299


SEED, SIDE, UPDATES = 20261012300, 192, 1200
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/EXACT_POINT_RANKER_300_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt'
PREVIOUS = ROOT / 'runs/oriented_slab_ranker_299_train_inner_001'
RUN = ROOT / 'runs/exact_point_ranker_300_train_inner_001'
EXPECTED_PARENT = 'b4689e1d0dcbf1e8d65a63a400962bf27f44b2e571c8361e29e0e51412b2b97f'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(PARENT) == EXPECTED_PARENT
prior_receipt = json.loads((PREVIOUS / 'completed.json').read_text())
assert prior_receipt['candidate'] == 299
assert sha(PREVIOUS / 'inner_panel.pt') == prior_receipt['output_sha256']['inner_panel.pt']
assert sha(PREVIOUS / 'draws.jsonl') == prior_receipt['output_sha256']['draws.jsonl']
assert sha(PREVIOUS / 'readout.jsonl') == prior_receipt['output_sha256']['readout.jsonl']
panel = torch.load(PREVIOUS / 'inner_panel.pt', map_location='cpu', weights_only=False)
prior_rows = [json.loads(line) for line in (PREVIOUS / 'readout.jsonl').open()]
used_before = {row['physical_section_id'] for row in
    (json.loads(line) for line in (PREVIOUS / 'draws.jsonl').open()) if row['accepted']}
assert len(panel) == len(prior_rows) == 64 and len(used_before) == 1200
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
rng = np.random.default_rng(SEED + 1)
config = {'candidate': 300, 'seed': SEED, 'updates': UPDATES,
    'train_bases': [0, 55], 'reused_adaptive_train_inner_panel_bases': [56, 63],
    'previous_completed_sha256': sha(PREVIOUS / 'completed.json'),
    'previous_panel_sha256': sha(PREVIOUS / 'inner_panel.pt'),
    'previous_draws_sha256': sha(PREVIOUS / 'draws.jsonl'),
    'parent_checkpoint_sha256': EXPECTED_PARENT,
    'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'train_correspondence_ranker_300.py', 'oriented_slab_correspondence_300.py',
        'oriented_slab_ranker_299.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_postartifact_gauge_v1.py', 'arbitrary_plane_one_shot_model.py')},
    'synthetic_provenance': context['provenance'],
    'rankers_random_initialized_identically': True, 'frozen_proposer': True,
    'external_weights_features_or_pseudolabels': False,
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False,
    'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

started, accepted, attempted = time.perf_counter(), 0, 0
with (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws, \
        (RUN / 'training.jsonl').open('w', encoding='utf8') as training:
    while accepted < UPDATES:
        assert time.perf_counter() - started < 7200, 'predeclared two-hour hard stop'
        attempted += 1
        base, variant = int(rng.integers(56)), int(rng.integers(VARIANTS))
        seed = SEED + 2_000_000_000 + attempted
        subject = base * VARIANTS + variant
        with torch.no_grad():
            sample = sample_one_shot_slide_artifacts_v9(context, [subject], seed, SIDE)
            if bool(sample['eligible'][0]):
                sample = sample_postartifact_gauge_v1(context, [subject], seed, SIDE,
                                                      source_sample=sample)
        identity = sample['provenance'][0]
        assert identity['physical_section_id'] not in used_before
        pair, error, central_support = None, None, None
        if bool(sample['eligible'][0]):
            with torch.no_grad():
                prediction = proposer.predict(sample['inputs'])
                all_prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                order = all_prior.argsort(descending=True)[:16]
                states, reflections, prior = prediction['state'][0, order // 2], order % 2, all_prior[order]
                pixels = sample['valid_mask'][0].flatten().nonzero().flatten()
                pixels = pixels[torch.linspace(0, len(pixels) - 1, min(256, len(pixels)),
                    device=pixels.device).long()]
                chart = torch.stack((pixels.remainder(SIDE), pixels.div(SIDE,
                    rounding_mode='floor')), -1).float() / SIDE
                truth = sample['centre'][0].reshape(-1, 3)[pixels]
                points = rigid_points_090(states[None], reflections[None], chart[None],
                                          (SIDE, SIDE))[0]
                error = (points - truth[None]).norm(-1).mean(-1) / 1000
                centre_bank = render_bank_299(context['atlas'], states, reflections,
                    sample['offsets'].expand(16, -1), sample['weights'].expand(16, -1), (0.,))
                central_support = centre_bank[:, 0, 1].mean((1, 2))
                near = int(error.argmin())
                allowed = ((error >= max(2., float(error[near]) + 1.))
                           & ((central_support - central_support[near]).abs() <= .10))
                if error[near] <= 1.5 and bool(allowed.any()):
                    far = int(torch.where(allowed, prior, -torch.inf).argmax())
                    pair = (near, far)
        draws.write(json.dumps({'attempt': attempted, 'accepted': pair is not None,
            'base': base, 'variant': variant, 'seed': seed,
            'physical_section_id': identity['physical_section_id'],
            'provenance': identity, 'near_far': pair,
            'candidate_error_mm': error.tolist() if error is not None else None,
            'candidate_support': central_support.tolist() if central_support is not None else None},
            allow_nan=False) + '\n')
        if pair is None:
            del sample
            continue
        accepted += 1
        near, far = pair
        extras = [index for index in range(16) if index not in pair]
        random_far = extras[int(rng.integers(len(extras)))]
        chosen = list(dict.fromkeys((near, far, 0, random_far)))
        with torch.no_grad():
            bank = render_bank_299(context['atlas'], states[chosen], reflections[chosen],
                sample['offsets'].expand(len(chosen), -1),
                sample['weights'].expand(len(chosen), -1))
            true_bank = render_bank_299(context['atlas'], sample['state'],
                sample['reflection'], sample['offsets'], sample['weights'])
        generator = torch.Generator(device='cuda').manual_seed(SEED + attempted)
        negatives = torch.randint(-12, 13, (64, 12, 3), device='cuda', generator=generator).float()
        negatives[..., 2] = torch.randint(-2, 3, (64, 12), device='cuda', generator=generator)
        too_close = negatives[..., :2].square().sum(-1) < 16
        negatives[..., 0] = torch.where(too_close,
            negatives[..., 0] + torch.where(negatives[..., 0] >= 0, 5., -5.),
            negatives[..., 0])
        target = F.avg_pool2d((sample['valid_mask'] &
            ~sample['fragment_displaced_mask']).float()[:, None], 4)
        active = [(chosen.index(index), index) for index in chosen if
                  index != near and float(error[index] - error[near]) >= .75 and
                  abs(float(central_support[index] - central_support[near])) <= .12]
        losses = {}
        for name, model in rankers.items():
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            evidence, reliability = score_bank_299(model, sample['inputs'], bank,
                                                    name == 'support_only')
            near_pos = chosen.index(near)
            pair_loss = torch.stack([
                F.softplus(4 * (.15 - (evidence[near_pos] - evidence[position])))
                + F.softplus(2 * (.25 - (prior[near] - prior[index]
                    + 4 * (evidence[near_pos] - evidence[position]))))
                for position, index in active]).mean()
            point_loss, usable = point_loss_300(model, sample, true_bank,
                                                negatives, name == 'support_only')
            tissue_loss = F.binary_cross_entropy_with_logits(reliability,
                                                              (target > .7).float())
            loss = pair_loss + .5 * point_loss + .1 * tissue_loss
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizers[name].step()
            losses[name] = {'pair': float(pair_loss.detach()),
                'point': float(point_loss.detach()), 'usable_points': usable,
                'reliability': float(tissue_loss.detach()),
                'gradient_norm': float(gradient.detach())}
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
        del sample, bank, true_bank

for model in rankers.values():
    model.eval()
rows = []
with (RUN / 'readout.jsonl').open('w', encoding='utf8') as output:
    with torch.inference_mode():
        for item, previous in zip(panel, prior_rows):
            assert item['physical_section_id'] == previous['physical_section_id']
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
                'prior_error_mm': float(error[0]), 'oracle_error_mm': float(error.min()),
                'previous_299_full_error_mm': previous['arms']['full']['selected_error_mm'],
                'candidate_error_mm': error.tolist(), 'candidate_prior_log': prior.tolist(),
                'near_far': near_far, 'arms': arms}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            if len(rows) % 16 == 0:
                print(json.dumps({'event': 'readout_milestone', 'cases': len(rows)}), flush=True)

eligible = [row for row in rows if row['near_far'] is not None]
summary = {'n': len(rows), 'matched_pairs': len(eligible),
    'prior_mean_mm': float(np.mean([row['prior_error_mm'] for row in rows])),
    'oracle_mean_mm': float(np.mean([row['oracle_error_mm'] for row in rows])),
    'previous_299_full_mean_mm': float(np.mean([row['previous_299_full_error_mm'] for row in rows]))}
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
    'full_selected_gain_over_299_at_least_0p15mm':
        summary['previous_299_full_mean_mm'] - summary['full']['mean_mm'] >= .15,
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
receipt = {'candidate': 300, 'attempted': attempted, 'accepted': accepted,
    'summary': summary,
    'checkpoint_sha256': {name: sha(RUN / f'{name}.pt') for name in rankers},
    'output_sha256': {name: sha(RUN / name) for name in (
        'config.json', 'draws.jsonl', 'training.jsonl', 'readout.jsonl', 'summary.json')},
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False,
    'elapsed_seconds': time.perf_counter() - started}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
