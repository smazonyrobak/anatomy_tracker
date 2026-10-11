"""TRAIN-only matched full-intensity versus complete-intensity-null 178 pilot."""

import hashlib
import json
import math
import os
import sys
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match
from training.spatial_blind_verifier_178 import SpatialBlindVerifier178


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def beam_for(prediction):
    # Identical to the frozen 176 8-base/6-anchor/2-normal-diverse rule.
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(len(beam), device=beam.device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(len(beam), 64, 2)[
            torch.arange(len(beam), device=beam.device), anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior, beam


def physical_error(states, reflection, reference):
    return (rigid_points_090(states, reflection, five) - reference[:, None]
            ).norm(dim=-1).mean(-1) / 1000


def draw_case(base, draw_seed, appearance, partition, attempt):
    sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
               else sample_one_shot_slide_artifacts_v4)
    with torch.no_grad():
        geometry = sample_panel_geometry_train_166(context, [base], draw_seed, 256)
        sample = sampler(context, [base], draw_seed, 256, source_sample=geometry)
        prediction = parent_model.predict(sample['inputs'])
        prior, beam = beam_for(prediction)
        ids = beam[0]
        states = prediction['state'][:, ids // 2]
        reflection = (ids % 2)[None]
        reference = rigid_points_090(sample['state'], sample['reflection'], five)
        initial_error = physical_error(states, reflection, reference)
    provenance = sample['provenance'][0]
    physical_id = provenance['physical_section_id']
    assert provenance['split'] == 'train' and physical_id not in physical_ids
    physical_ids.add(physical_id)
    normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
    angle = math.degrees(math.acos(float(normal.abs().max().clamp(-1, 1))))
    record = {'partition': partition, 'attempt': attempt, 'appearance': appearance,
        'base_index': base, 'draw_seed': draw_seed, 'physical_section_id': physical_id,
        'eligible': bool(sample['eligible'][0]), 'used': False,
        'valid_pixels': int(sample['valid_mask'][0].sum()),
        'nearest_cardinal_angle_deg': angle,
        'model_inputs_sha256': hashlib.sha256(sample['inputs'].detach().cpu().contiguous()
            .numpy().tobytes()).hexdigest(),
        'beam_branch_ids': ids.tolist(), 'beam_prior_logits': prior[0, ids].tolist(),
        'beam_state12': states[0].tolist(), 'beam_reflection': reflection[0].tolist(),
        'beam_original_five_point_mm': initial_error[0].tolist(),
        'provenance': provenance}
    return sample, prediction, prior, beam, reference, record


def frozen_case(sample, prediction, prior, beam, reference, record):
    ids = beam[0]
    modes = ids // 2
    flags = (ids % 2)[None]
    initial = prediction['state'][:, modes]
    original_error = physical_error(initial, flags, reference)[0]
    native_scores, native_states = [], []
    with torch.no_grad():
        for start in range(0, 16, 4):
            native = global_plane_match(parent_model, prediction, modes[None, start:start + 4],
                flags[:, start:start + 4], sample['offsets'], sample['weights'], atlas,
                (256, 256), side=24)
            native_scores.append(torch.stack((native['input_score'], native['score']), -1))
            native_states.append(native['state'])
        native_score = torch.cat(native_scores, 1)[0]
        native_state = torch.cat(native_states, 1)
        native_error = physical_error(native_state, flags, reference)[0]
    direct_id = int(prior[0].argmax())
    assert direct_id in ids.tolist()
    direct_slot = ids.tolist().index(direct_id)
    native_choice = int(native_score.flatten().argmax())
    native_slot, native_action = divmod(native_choice, 2)
    case = {'record': record, 'inputs': sample['inputs'].cpu(),
        'offsets': sample['offsets'].cpu(), 'weights': sample['weights'].cpu(),
        'state': initial.cpu(), 'reflection': flags.cpu(),
        'prior': prior[:, ids].cpu(), 'reference': reference.cpu(),
        'truth_reflection': sample['reflection'].cpu(),
        'original_error': original_error.cpu(),
        'direct_branch_id': direct_id, 'direct_mm': float(original_error[direct_slot]),
        'native_branch_id': int(ids[native_slot]),
        'native_original_mm': float(original_error[native_slot]),
        'native_action_mm': float(original_error[native_slot] if native_action == 0
                                  else native_error[native_slot]),
        'native_action': native_action, 'native_action_logits32': native_score.flatten().tolist()}
    return case


def keys_for(draw_seed):
    generator = torch.Generator().manual_seed(draw_seed)
    return torch.rand((1, 16, 96 * 96), generator=generator).cuda()


def evaluate_case(case, name, verifier, step, cohort):
    inputs = case['inputs'].cuda()
    states = case['state'].cuda()
    reflection = case['reflection'].cuda()
    prior = case['prior'].cuda()
    reference = case['reference'].cuda()
    with torch.no_grad():
        output = verifier.actions(inputs, states, reflection, case['offsets'].cuda(),
            case['weights'].cuda(), atlas, prior,
            keys_for(case['record']['draw_seed']) if name == 'null' else None)
        original = case['original_error'].cuda()
        corrected = physical_error(output['state'], reflection, reference)[0]
        scores = output['logits'][0]
        original_slot = int(scores[:, 0].argmax())
        choice = int(scores.flatten().argmax())
        action_slot, action = divmod(choice, 2)
        support_mask = (output['original_support'][0] > .5).flatten(1).float()
        support = support_mask.sum(-1)
        near = ((reflection[0] == case['truth_reflection'].cuda()[0])
                & (original >= .5) & (original <= 3.))
        difference = original[:, None] - original[None, :]
        intersection = support_mask @ support_mask.T
        support_iou = intersection / (support[:, None] + support[None, :] - intersection
                                      ).clamp_min(1)
        matched = (torch.triu(torch.ones((16, 16), device='cuda', dtype=torch.bool), 1)
                   & (support_iou >= .7)
                   & (difference.abs() >= .25))
        ordered = ((scores[:, 0, None] - scores[None, :, 0]) * -difference > 0) & matched
        row = {'step': step, 'cohort': cohort, 'arm': name,
            'physical_section_id': case['record']['physical_section_id'],
            'appearance': case['record']['appearance'],
            'appearance_mode': case['record']['provenance']['mode'],
            'base_index': case['record']['base_index'],
            'valid_pixels': case['record']['valid_pixels'],
            'nearest_cardinal_angle_deg': case['record']['nearest_cardinal_angle_deg'],
            'beam_branch_ids': case['record']['beam_branch_ids'],
            'original_five_point_mm16': original.tolist(),
            'corrected_five_point_mm16': corrected.tolist(),
            'original_support_sites16': support.int().tolist(),
            'corrected_support_sites16': (output['corrected_support'][0] > .5
                ).sum((-2, -1)).int().tolist(),
            'support_retention16': ((output['corrected_support'][0]
                * (output['original_support'][0] > .5)).sum((-2, -1))
                / (output['original_support'][0]
                   * (output['original_support'][0] > .5)).sum((-2, -1)).clamp_min(1)).tolist(),
            'action_logits32': scores.flatten().tolist(),
            'prior_weight': float(output['prior_weight']),
            'mean_q_norm': float(output['q'][0].norm(dim=-1).mean()),
            'selected_original_branch_id': case['record']['beam_branch_ids'][original_slot],
            'selected_original_mm': float(original[original_slot]),
            'selected_action_branch_id': case['record']['beam_branch_ids'][action_slot],
            'selected_action': 'original' if action == 0 else 'corrected',
            'selected_action_mm': float(original[action_slot] if action == 0
                                        else corrected[action_slot]),
            'direct_branch_id': case['direct_branch_id'], 'direct_mm': case['direct_mm'],
            'native_branch_id': case['native_branch_id'],
            'native_original_mm': case['native_original_mm'],
            'native_action_mm': case['native_action_mm'],
            'native_action': case['native_action'],
            'near_branch_count': int(near.sum()),
            'near_start_sum_mm': float(original[near].sum()),
            'near_corrected_sum_mm': float(corrected[near].sum()),
            'near_improved_count': int((corrected[near] < original[near]).sum()),
            'support_matched_pairs': int(matched.sum()),
            'support_matched_correct': int(ordered.sum())}
    return row


def evaluate_cases(cases, step, cohort, destination):
    rows = []
    with destination.open('w') as stream:
        for case in cases:
            for name in ('full', 'null'):
                row = evaluate_case(case, name, verifiers[name], step, cohort)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    return rows


def summarise(rows):
    metrics = ('direct_mm', 'native_original_mm', 'native_action_mm',
               'selected_original_mm', 'selected_action_mm')
    return {'sections': len(rows),
        'mean_mm': {key: float(np.mean([row[key] for row in rows])) if rows else None
                    for key in metrics},
        'capture_le_1p5': {key: float(np.mean([row[key] <= 1.5 for row in rows]))
                           if rows else None for key in metrics},
        'corrected_action_fraction': float(np.mean([row['selected_action'] == 'corrected'
            for row in rows])) if rows else None,
        'near_branch_count': sum(row['near_branch_count'] for row in rows),
        'near_start_mm': (sum(row['near_start_sum_mm'] for row in rows)
            / sum(row['near_branch_count'] for row in rows)
            if any(row['near_branch_count'] for row in rows) else None),
        'near_corrected_mm': (sum(row['near_corrected_sum_mm'] for row in rows)
            / sum(row['near_branch_count'] for row in rows)
            if any(row['near_branch_count'] for row in rows) else None),
        'near_improved_fraction': (sum(row['near_improved_count'] for row in rows)
            / sum(row['near_branch_count'] for row in rows)
            if any(row['near_branch_count'] for row in rows) else None),
        'support_matched_pairs': sum(row['support_matched_pairs'] for row in rows),
        'support_matched_ordering': (sum(row['support_matched_correct'] for row in rows)
            / sum(row['support_matched_pairs'] for row in rows)
            if any(row['support_matched_pairs'] for row in rows) else None),
        'mean_support_retention': float(np.mean([value for row in rows
            for value in row['support_retention16']])) if rows else None,
        'prior_weight': float(np.mean([row['prior_weight'] for row in rows])) if rows else None}


def report(rows, step, cohort):
    result = {'step': step, 'cohort': cohort, 'by_appearance': {}, 'gate': {}}
    for appearance in ('v3', 'v4'):
        result['by_appearance'][appearance] = {}
        selected = [row for row in rows if row['appearance'] == appearance]
        threshold = float(np.quantile([row['valid_pixels'] for row in selected], .25))
        for name in ('full', 'null'):
            part = [row for row in selected if row['arm'] == name]
            result['by_appearance'][appearance][name] = {
                'population': summarise(part),
                'by_base': {str(base): summarise([row for row in part
                    if row['base_index'] == base]) for base in sorted({row['base_index']
                    for row in part})},
                'by_appearance_mode': {mode: summarise([row for row in part
                    if row['appearance_mode'] == mode]) for mode in
                    ('raw', 'exact_black', 'imperfect_brush')},
                'sparse_valid_pixel_q25': threshold,
                'sparse_tissue': summarise([row for row in part
                    if row['valid_pixels'] <= threshold]),
                'steep_oblique_ge_30deg': summarise([row for row in part
                    if row['nearest_cardinal_angle_deg'] >= 30])}
        full = result['by_appearance'][appearance]['full']['population']
        null = result['by_appearance'][appearance]['null']['population']
        result['gate'][appearance] = {
            'original_mean_below_direct_by_0p25': full['mean_mm']['selected_original_mm']
                <= full['mean_mm']['direct_mm'] - .25,
            'original_mean_below_native_branch_by_0p25': full['mean_mm']['selected_original_mm']
                <= full['mean_mm']['native_original_mm'] - .25,
            'original_capture_above_direct_by_0p10': full['capture_le_1p5']['selected_original_mm']
                >= full['capture_le_1p5']['direct_mm'] + .10,
            'original_capture_above_native_branch_by_0p10': full['capture_le_1p5']['selected_original_mm']
                >= full['capture_le_1p5']['native_original_mm'] + .10,
            'original_mean_below_complete_null_by_0p10': full['mean_mm']['selected_original_mm']
                <= null['mean_mm']['selected_original_mm'] - .10,
            'selected_action_not_worse': full['mean_mm']['selected_action_mm']
                <= full['mean_mm']['selected_original_mm']}
    result['continuation_gate_passes'] = all(all(checks.values())
        for checks in result['gate'].values())
    result['selection_value_mm'] = float(np.mean([result['by_appearance'][appearance]
        ['full']['population']['mean_mm']['selected_original_mm'] for appearance in ('v3', 'v4')]))
    return result


def fresh_cases(step):
    cases = []
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, 178]))
    with (run / f'fresh_draws_{step:05d}.jsonl').open('w') as stream:
        for appearance in ('v3', 'v4'):
            for base in rng.permutation(training_bases)[:16]:
                while True:
                    draw_seed = int(rng.integers(0, 2**63 - 1))
                    sample, prediction, prior, beam, reference, record = draw_case(
                        int(base), draw_seed, appearance, 'fresh_check', len(physical_ids))
                    record['used'] = record['eligible']
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    if record['eligible']:
                        cases.append(frozen_case(sample, prediction, prior, beam, reference, record))
                        break
    return cases


def save_checkpoint(step, best, history):
    path = run / f'verifier_step_{step:05d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save({'step': step, 'full': verifiers['full'].state_dict(),
        'null': verifiers['null'].state_dict(),
        'full_optimizer': optimizers['full'].state_dict(),
        'null_optimizer': optimizers['null'].state_dict(),
        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'physical_ids': sorted(physical_ids),
        'best': best, 'history': history, 'config': config, 'calibrated': False}, temporary)
    os.replace(temporary, path)
    return path


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/SPATIAL_BLIND_VERIFIER_178_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
run = Path(os.environ.get('AT178_RUN', str(root / 'runs/spatial_blind_verifier_178_train_001'))).resolve()
resume = os.environ.get('AT178_RESUME')
max_updates = int(os.environ.get('AT178_STEPS', '5000'))
seed = 202610178
panel_seed = 202610179
panel_bases = (3, 11, 19, 27, 35, 43, 51, 59)
training_bases = tuple(base for base in range(64) if base not in panel_bases)
milestones = (0, 1000, 2500, 5000)
assert len(training_bases) == 56 and len(panel_bases) == 8
assert source.drive.upper() == run.drive.upper() == root.drive.upper() == 'I:'
assert max_updates in (1000, 2500, 5000) and not run.exists() and torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
parent_model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(parent_model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    frozen = torch.load(parent_path, map_location='cpu', weights_only=True)
assert frozen['step'] == 2000 and frozen['calibrated'] is False
parent_model.load_state_dict(frozen['model'], strict=True)
del frozen
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
verifiers = {'full': SpatialBlindVerifier178().cuda(),
             'null': SpatialBlindVerifier178().cuda()}
verifiers['null'].load_state_dict(verifiers['full'].state_dict(), strict=True)
optimizers = {name: torch.optim.AdamW(verifier.parameters(), lr=2e-4, weight_decay=1e-4)
              for name, verifier in verifiers.items()}
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / 256
rng = np.random.default_rng(seed)
physical_ids = set()
resume_sha, start_step, best, history = None, 0, None, {}
resume_ids, resume_rng, resume_torch_rng, resume_cuda_rng = None, None, None, None
if resume:
    resume_path = Path(resume).resolve()
    assert resume_path.drive.upper() == 'I:'
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
    start_step = int(saved['step'])
    assert start_step in milestones and start_step < max_updates
    assert saved['config']['protocol_sha256'] == sha(protocol)
    assert saved['config']['parent_checkpoint_sha256'] == sha(parent_path)
    assert saved['config']['source_sha256']['spatial_blind_verifier_178.py'] == sha(
        source / 'spatial_blind_verifier_178.py')
    for name in verifiers:
        verifiers[name].load_state_dict(saved[name], strict=True)
        optimizers[name].load_state_dict(saved[f'{name}_optimizer'])
    resume_rng = saved['numpy_rng']
    resume_ids = set(saved['physical_ids'])
    best, history = saved['best'], saved['history']
    resume_sha = sha(resume_path)
    resume_torch_rng, resume_cuda_rng = saved['torch_rng'], saved['cuda_rng']
    resume_source_sha256 = saved['config']['source_sha256']
    del saved

source_files = ('train_spatial_blind_verifier_178.py', 'spatial_blind_verifier_178.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py', 'arbitrary_plane_panel_geometry_train_166.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_full_frame_primitives.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
if resume:
    assert all(sha(source / name) == digest for name, digest in resume_source_sha256.items())
config = {'seed': seed, 'panel_seed': panel_seed,
    'start_update': start_step, 'requested_end_update': max_updates,
    'resume_checkpoint': str(resume_path) if resume else None,
    'resume_checkpoint_sha256': resume_sha,
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'panel_bases': panel_bases, 'gradient_bases': training_bases,
    'panel': '8 independent eligible full-sphere TRAIN-166 planes per base and appearance',
    'fresh_check': '16 independently drawn eligible gradient-base planes per appearance at each later milestone',
    'train_batch': 'one accepted independent TRAIN-166 section, alternating v3/v4; same draw and beam in both arms',
    'null': 'fixed random within-current-support permutation of atlas intensity at original and corrected 96px renders; support unchanged',
    'pairwise_ordering': 'original-branch pairs with atlas-support IoU >= 0.7 and physical-error gap >= 0.25 mm',
    'actions': '16 original plus 16 bounded 9D corrected rigid planes; original action always available',
    'loss': 'listwise KL to truth-derived physical five-point 32-action costs / 0.7 plus 0.25 near matching-reflection correction mm plus 0.01 q^2 and 0.05 initial-support retention penalties',
    'checkpoint_selection': 'minimum equal-v3/v4 mean selected-original five-point error on fixed inner TRAIN panel',
    'continuation_stop': 'only after both 1000 and 2500 full-arm fixed-panel and independent fresh-check original selections are worse than the unchanged direct prior in both appearances',
    'gate_native_comparator': 'original rigid branch chosen by frozen 121 32-action fit score; fitted action separately reported',
    'real_training_images': 0, 'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'development_used': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False}
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

panel_cases = []
panel_rng = np.random.default_rng(panel_seed)
with (run / 'panel_draws.jsonl').open('w') as stream:
    for appearance in ('v3', 'v4'):
        for base in panel_bases:
            for _ in range(8):
                while True:
                    draw_seed = int(panel_rng.integers(0, 2**63 - 1))
                    sample, prediction, prior, beam, reference, record = draw_case(
                        base, draw_seed, appearance, 'fixed_inner_train', len(physical_ids))
                    record['used'] = record['eligible']
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    if record['eligible']:
                        panel_cases.append(frozen_case(sample, prediction, prior, beam,
                                                       reference, record))
                        break
assert len(panel_cases) == 128
assert all(sum(case['record']['appearance'] == appearance and
    case['record']['base_index'] == base for case in panel_cases) == 8
    for appearance in ('v3', 'v4') for base in panel_bases)

if resume:
    # Panel reconstruction is deterministic and does not alter the saved training RNG stream.
    assert physical_ids <= resume_ids
    physical_ids = resume_ids
    rng.bit_generator.state = resume_rng
    torch.set_rng_state(resume_torch_rng)
    torch.cuda.set_rng_state_all(resume_cuda_rng)
else:
    for verifier in verifiers.values():
        verifier.eval()
    panel_rows = evaluate_cases(panel_cases, 0, 'fixed_inner_train', run / 'panel_00000.jsonl')
    panel_report = report(panel_rows, 0, 'fixed_inner_train')
    (run / 'panel_summary_00000.json').write_text(json.dumps(panel_report, indent=2,
        allow_nan=False))
    best = {'step': 0, 'selection_value_mm': panel_report['selection_value_mm'],
            'checkpoint': str(run / 'verifier_step_00000.pt')}
    history['0'] = {'panel': panel_report, 'fresh': None}
    save_checkpoint(0, best, history)

stop_reason = None
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as training:
    for step in range(start_step + 1, max_updates + 1):
        appearance = 'v3' if step % 2 else 'v4'
        while True:
            base = int(rng.choice(training_bases))
            draw_seed = int(rng.integers(0, 2**63 - 1))
            sample, prediction, prior, beam, reference, record = draw_case(
                base, draw_seed, appearance, 'gradient', len(physical_ids))
            record['update'] = step
            record['used'] = record['eligible']
            draws.write(json.dumps(record, allow_nan=False) + '\n')
            if record['eligible']:
                break
        ids = beam[0]
        states = prediction['state'][:, ids // 2].detach()
        reflection = (ids % 2)[None]
        candidate_prior = prior[:, ids].detach()
        original_error = physical_error(states, reflection, reference).detach()
        near = ((reflection == sample['reflection'][:, None])
                & (original_error >= .5) & (original_error <= 3.))
        keys = keys_for(draw_seed)
        detail = {'update': step, 'appearance': appearance, 'base_index': base,
                  'draw_seed': draw_seed, 'physical_section_id': record['physical_section_id'],
                  'near_branches': int(near.sum()), 'arms': {}}
        for name, verifier in verifiers.items():
            verifier.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            output = verifier.actions(sample['inputs'], states, reflection,
                sample['offsets'], sample['weights'], atlas, candidate_prior,
                keys if name == 'null' else None)
            corrected_error = physical_error(output['state'], reflection, reference)
            costs = torch.stack((original_error, corrected_error), -1)
            target = F.softmax(-costs.detach().flatten(1) / .7, -1)
            ranking = F.kl_div(F.log_softmax(output['logits'].flatten(1), -1),
                target, reduction='batchmean')
            correction = corrected_error[near].mean() if bool(near.any()) else ranking.new_zeros(())
            initial_mask = (output['original_support'] > .5).to(states.dtype)
            retention = ((output['corrected_support'] * initial_mask).sum((-2, -1))
                         / (output['original_support'] * initial_mask
                            ).sum((-2, -1)).clamp_min(1))
            q_penalty = output['q'].square().sum(-1).mean()
            support_penalty = (1 - retention).clamp_min(0).mean()
            loss = ranking + .25 * correction + .01 * q_penalty + .05 * support_penalty
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(verifier.parameters(), 5.,
                error_if_nonfinite=True)
            lr = 2e-4 * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / 5000)))
            optimizer.param_groups[0]['lr'] = lr
            optimizer.step()
            detail['arms'][name] = {'loss': float(loss.detach()),
                'rank_loss': float(ranking.detach()), 'near_correction_mm': float(correction.detach()),
                'q_penalty': float(q_penalty.detach()),
                'support_penalty': float(support_penalty.detach()),
                'gradient_norm': float(gradient), 'prior_weight': float(output['prior_weight'].detach())}
        training.write(json.dumps(detail, allow_nan=False) + '\n')
        if step % 50 == 0:
            draws.flush()
            training.flush()
        if step in milestones[1:]:
            draws.flush()
            training.flush()
            for verifier in verifiers.values():
                verifier.eval()
            panel_rows = evaluate_cases(panel_cases, step, 'fixed_inner_train',
                run / f'panel_{step:05d}.jsonl')
            panel_report = report(panel_rows, step, 'fixed_inner_train')
            (run / f'panel_summary_{step:05d}.json').write_text(json.dumps(
                panel_report, indent=2, allow_nan=False))
            checks = fresh_cases(step)
            fresh_rows = evaluate_cases(checks, step, 'fresh_gradient_bases',
                run / f'fresh_{step:05d}.jsonl')
            fresh_report = report(fresh_rows, step, 'fresh_gradient_bases')
            (run / f'fresh_summary_{step:05d}.json').write_text(json.dumps(
                fresh_report, indent=2, allow_nan=False))
            history[str(step)] = {'panel': panel_report, 'fresh': fresh_report}
            if panel_report['selection_value_mm'] < best['selection_value_mm']:
                best = {'step': step, 'selection_value_mm': panel_report['selection_value_mm'],
                        'checkpoint': str(run / f'verifier_step_{step:05d}.pt')}
            checkpoint = save_checkpoint(step, best, history)
            print(json.dumps({'event': 'milestone', 'step': step,
                'checkpoint_sha256': sha(checkpoint), 'best_step': best['step'],
                'panel_original_mm': panel_report['selection_value_mm'],
                'gate_passes': panel_report['continuation_gate_passes']}), flush=True)
            if step == 2500:
                sustained = all(
                    history[str(milestone)][cohort]['by_appearance'][appearance]['full']
                        ['population']['mean_mm']['selected_original_mm']
                    > history[str(milestone)][cohort]['by_appearance'][appearance]['full']
                        ['population']['mean_mm']['direct_mm']
                    for milestone in (1000, 2500) for cohort in ('panel', 'fresh')
                    for appearance in ('v3', 'v4'))
                if sustained:
                    stop_reason = 'sustained_panel_and_fresh_deterioration_at_1000_and_2500'
                    break

last_step = step
best_path = Path(best['checkpoint'])
assert best_path.drive.upper() == 'I:' and best_path.exists()
(run / 'best_checkpoint.json').write_text(json.dumps({**best,
    'checkpoint_sha256': sha(best_path), 'selection_cohort': 'fixed_inner_train',
    'metric': 'equal-v3/v4 mean selected original branch five-point mm'},
    indent=2, allow_nan=False))
(run / 'completed.json').write_text(json.dumps({'start_update': start_step,
    'actual_end_update': last_step, 'requested_end_update': max_updates,
    'stop_reason': stop_reason or ('operator_cap' if last_step < 5000 else '5000_cap'),
    'best_step': best['step'], 'best_checkpoint_sha256': sha(best_path),
    'accepted_gradient_sections': last_step - start_step,
    'panel_sections': len(panel_cases), 'panel_bases': panel_bases,
    'gradient_bases': training_bases,
    'config_sha256': sha(run / 'config.json'),
    'output_sha256': {path.name: sha(path) for path in run.iterdir()
        if path.is_file() and path.name != 'completed.json'},
    'calibrated': False, 'real_training_images': 0, 'development_used': False,
    'final_animals_used': False, 'public_benchmark_used': False},
    indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'actual_updates': last_step,
    'best_step': best['step'], 'stop_reason': stop_reason}), flush=True)
