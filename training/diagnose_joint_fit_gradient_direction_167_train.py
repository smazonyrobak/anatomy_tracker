"""Frozen TRAIN-only local direction test of the measured 162 fit objective."""

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

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.joint_pose_map_121 import joint_fit_loss_121, joint_forward_121


assert os.environ.get('JOINT167_APPROVED') == '1', 'the bounded GPU diagnostic awaits approval'
source = Path(__file__).resolve().parent
run = root / 'runs/joint_fit_gradient_direction_167_train_001'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_checkpoint = parent_dir / 'joint_step_02000.pt'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
seed, attempts, side, step_q = 202610111671, 128, 256, .5
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
q_scale = torch.tensor((math.pi / 36,) * 3 + (500.,) * 3 + (.05,) * 3,
                       device='cuda', dtype=torch.float32)
source_names = ('diagnose_joint_fit_gradient_direction_167_train.py',
    'arbitrary_plane_panel_geometry_train_166.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_recurrent_model.py', 'arbitrary_plane_ribbon_v6.py',
    'whole_slice_atlas_feedback_083.py',
    'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py', 'global_atlas_contrast_090.py',
    'global_plane_matcher_120.py', 'joint_pose_map_121.py')
source_sha = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in source_names}
with parent_checkpoint.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert parent_sha == parent_done['checkpoint_sha256']['2000']
assert source_sha['arbitrary_plane_one_shot_model.py'] == parent_config['source_sha256'][
    'arbitrary_plane_one_shot_model.py']
assert source_sha['global_plane_matcher_120.py'] == parent_config['source_sha256'][
    'global_plane_matcher_120.py']
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda()
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
model.eval().requires_grad_(False)
del saved

schedule = [{'draw_index': i, 'base_index': (37 * i) % 60,
             'draw_seed': seed + 1009 * i,
             'random_seed': seed + 3001 * i + 7}
            for i in range(attempts)]
config = {'scope': 'TRAIN-only, synthetic physical sections, no fitting or training',
    'source_sha256': source_sha, 'parent_checkpoint_sha256': parent_sha,
    'parent_step': 2000, 'synthetic_provenance': context['provenance'],
    'draw_count': attempts, 'draw_seed': seed,
    'base_schedule': '(37 * draw_index) % 60, no validity retry',
    'blind_selection': 'maximum pre-fit pose log mass over 160 mode/reflection branches',
    'near_stratum_after_scoring': 'baseline corrected rigid error <=1.5 mm; never used for selection',
    'tissue_weight': 'synthetic valid_mask as in 162; unavailable at inference',
    'measured_objective': 'LNCC fit + 0.5 coverage penalty + 0.05 warp + 0.05 correction',
    'q_coordinates': 'right-composed local SO(3) radians3, local translation um3, log-basis2, shear1',
    'q_scale': q_scale.tolist(), 'step_l2_q': step_q,
    'effectively_zero_gradient_norm_q': 1e-12,
    'directions': 'negative normalized measured gradient, seeded random unit vector, negative normalized truth rigid gradient',
    'rigid_score': 'mean CCF Euclidean distance in mm over all valid observed pixel sites, after learned matcher correction; uncorrected also recorded',
    'no_truth_in_fit_direction': 'only synthetic tissue valid_mask enters measured objective; pose truth enters rigid scoring and oracle control only',
    'no_dev_or_public': True, 'no_model_promotion': True}
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
with (run / 'draws.jsonl').open('w') as stream:
    for row in schedule:
        stream.write(json.dumps(row, allow_nan=False) + '\n')

cases = []
with (run / 'cases.jsonl').open('w') as stream:
    for row in schedule:
        with torch.no_grad():
            original = sample_panel_geometry_train_166(context, [row['base_index']],
                                                        row['draw_seed'], side)
            sample = sample_one_shot_slide_artifacts_v4(context, [row['base_index']],
                row['draw_seed'], side, source_sample=original)
        provenance = sample['provenance'][0]
        valid = sample['valid_mask'][0]
        pixels = int(valid.sum())
        case = {**row, 'physical_section_id': provenance['physical_section_id'],
            'animal_id': provenance['base_lineage']['animal_id'],
            'provenance': provenance, 'valid_pixels': pixels,
            'valid_fraction': pixels / side**2, 'eligible': bool(sample['eligible'][0]),
            'blind_branch_id': None, 'baseline_input_rigid_mm': None,
            'baseline_corrected_rigid_mm': None, 'retrospective_near': None,
            'baseline_objective': None, 'baseline_atlas_coverage': None,
            'common_support_equivalent_64px': None,
            'component_values': None, 'component_gradient_norms_q': None,
            'measured_gradient_norm_q': None, 'oracle_gradient_norm_q': None,
            'measured_gradient_effectively_zero': None,
            'oracle_gradient_effectively_zero': None,
            'measured_oracle_gradient_cosine': None, 'directions': None,
            'skip_reason': None}
        if not pixels:
            case['skip_reason'] = 'zero_valid_brain_intersection'
            stream.write(json.dumps(case, allow_nan=False) + '\n')
            cases.append(case)
            continue
        with torch.no_grad():
            prediction = model.predict(sample['inputs'])
            prior = prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)
            branch = int(prior.flatten(1).argmax())
            mode = torch.tensor([[branch // 2]], device='cuda')
            reflection = torch.tensor([[branch % 2]], device='cuda')
            input_state = prediction['state'][:, branch // 2:branch // 2 + 1]
            ids = valid.flatten().nonzero().flatten()
            chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                                -1).float()[None] / side
            truth = rigid_points_090(sample['state'], sample['reflection'], chart)

        q = torch.zeros((1, 1, 9), device='cuda', requires_grad=True)
        candidate_state = compose_full_frame_state(input_state, q * q_scale)
        joint = joint_forward_121(model, prediction, sample['inputs'], mode, reflection,
            atlas, sample['offsets'], sample['weights'], candidate_state=candidate_state,
            match_side=24, map_side=96)
        appearance = joint_fit_loss_121(sample['inputs'], joint['mapped'], atlas,
            sample['weights'], sample['valid_mask'], side=64)
        local = joint['mapped']['local_displacement_um'].float() / 1000
        warp = (local.square().mean((2, 3, 4))
            + .3 * (local[..., 1:] - local[..., :-1]).square().mean((2, 3, 4))
            + .3 * (local[..., 1:, :] - local[..., :-1, :]).square().mean((2, 3, 4)))
        correction = (joint['match']['update'][..., 3:6] / 2000).square().sum(-1)
        components = {'fit': appearance['fit_loss'][0, 0],
            'coverage_half': .5 * appearance['coverage_penalty'][0, 0],
            'warp_005': .05 * warp[0, 0],
            'correction_005': .05 * correction[0, 0]}
        measured = sum(components.values())
        proposed = rigid_points_090(joint['mapped']['state'], reflection, chart[:, None])
        rigid = ((proposed - truth[:, None]).norm(dim=-1).mean() / 1000)
        component_grad = {}
        for name, value in components.items():
            gradient = torch.autograd.grad(value, q, retain_graph=True, allow_unused=True)[0]
            component_grad[name] = torch.zeros_like(q) if gradient is None else gradient.detach()
        measured_grad = torch.autograd.grad(measured, q, retain_graph=True)[0].detach()
        oracle_grad = torch.autograd.grad(rigid, q)[0].detach()
        if not bool(torch.isfinite(torch.stack((measured.detach(), rigid.detach(),
                measured_grad.norm(), oracle_grad.norm(),
                *[gradient.norm() for gradient in component_grad.values()]))).all()):
            case['skip_reason'] = 'nonfinite_objective_or_gradient'
            stream.write(json.dumps(case, allow_nan=False) + '\n')
            cases.append(case)
            continue
        measured_norm = float(measured_grad.norm())
        oracle_norm = float(oracle_grad.norm())
        case.update(blind_branch_id=branch,
            baseline_input_rigid_mm=float((rigid_points_090(input_state, reflection,
                chart[:, None]) - truth[:, None]).norm(dim=-1).mean() / 1000),
            baseline_corrected_rigid_mm=float(rigid.detach()),
            retrospective_near=bool(float(rigid.detach()) <= 1.5),
            baseline_objective=float(measured.detach()),
            baseline_atlas_coverage=float(appearance['atlas_coverage'][0, 0].detach()),
            common_support_equivalent_64px=float(
                appearance['atlas_coverage'][0, 0].detach()
                * F.interpolate(valid[None, None].float(), (64, 64), mode='area').sum()),
            component_values={name: float(value.detach()) for name, value in components.items()},
            component_gradient_norms_q={name: float(gradient.norm())
                for name, gradient in component_grad.items()},
            measured_gradient_norm_q=measured_norm,
            oracle_gradient_norm_q=oracle_norm,
            measured_gradient_effectively_zero=bool(measured_norm <= 1e-12),
            oracle_gradient_effectively_zero=bool(oracle_norm <= 1e-12),
            measured_oracle_gradient_cosine=(float((measured_grad * oracle_grad).sum()
                / (measured_norm * oracle_norm)) if measured_norm and oracle_norm else None))
        random = torch.from_numpy(np.random.default_rng(row['random_seed'])
            .standard_normal(9).astype('float32')).to('cuda').reshape(1, 1, 9)
        random = random / random.norm()
        directions = {'measured_descent': (-measured_grad / measured_grad.norm()
            if measured_norm > 1e-12 else torch.zeros_like(measured_grad)),
            'random': random, 'oracle_descent': (-oracle_grad / oracle_grad.norm()
            if oracle_norm > 1e-12 else torch.zeros_like(oracle_grad))}
        case['directions'] = {}
        del joint, appearance, local, warp, correction, components, measured, rigid, q
        with torch.no_grad():
            for name, direction in directions.items():
                state = compose_full_frame_state(input_state, step_q * direction * q_scale)
                stepped = joint_forward_121(model, prediction, sample['inputs'], mode,
                    reflection, atlas, sample['offsets'], sample['weights'],
                    candidate_state=state, match_side=24, map_side=96)
                fit = joint_fit_loss_121(sample['inputs'], stepped['mapped'], atlas,
                    sample['weights'], sample['valid_mask'], side=64)
                local = stepped['mapped']['local_displacement_um'].float() / 1000
                warp = (local.square().mean((2, 3, 4))
                    + .3 * (local[..., 1:] - local[..., :-1]).square().mean((2, 3, 4))
                    + .3 * (local[..., 1:, :] - local[..., :-1, :]).square().mean((2, 3, 4)))
                correction = (stepped['match']['update'][..., 3:6] / 2000).square().sum(-1)
                objective = (fit['fit_loss'] + .5 * fit['coverage_penalty']
                    + .05 * warp + .05 * correction)[0, 0]
                proposed = rigid_points_090(stepped['mapped']['state'], reflection, chart[:, None])
                stepped_rigid = float((proposed - truth[:, None]).norm(dim=-1).mean() / 1000)
                input_rigid = float((rigid_points_090(state, reflection, chart[:, None])
                    - truth[:, None]).norm(dim=-1).mean() / 1000)
                case['directions'][name] = {'q_step_l2': float((step_q * direction).norm()),
                    'objective': float(objective),
                    'objective_delta': float(objective) - case['baseline_objective'],
                    'corrected_rigid_mm': stepped_rigid,
                    'corrected_rigid_delta_mm': stepped_rigid - case['baseline_corrected_rigid_mm'],
                    'input_rigid_mm': input_rigid,
                    'input_rigid_delta_mm': input_rigid - case['baseline_input_rigid_mm'],
                    'atlas_coverage': float(fit['atlas_coverage'][0, 0])}
        stream.write(json.dumps(case, allow_nan=False) + '\n')
        stream.flush()
        cases.append(case)
        if (row['draw_index'] + 1) % 32 == 0:
            print(json.dumps({'event': 'TRAIN_direction_milestone',
                'attempts': row['draw_index'] + 1, 'target_attempts': attempts}), flush=True)

scored = [case for case in cases if case['directions'] is not None]
groups = {'all_scored': scored,
    'retrospective_near_le_1p5mm': [case for case in scored if case['retrospective_near']],
    'retrospective_far_gt_1p5mm': [case for case in scored if not case['retrospective_near']]}
summary = {'attempt_n': len(cases),
    'zero_valid_n': sum(case['skip_reason'] == 'zero_valid_brain_intersection' for case in cases),
    'nonfinite_n': sum(case['skip_reason'] == 'nonfinite_objective_or_gradient' for case in cases),
    'zero_measured_gradient_n': sum(case['measured_gradient_effectively_zero'] is True
        for case in cases),
    'zero_or_nonfinite_measured_gradient_n': sum(case['measured_gradient_effectively_zero'] is True
        or case['skip_reason'] == 'nonfinite_objective_or_gradient' for case in cases),
    'scored_n': len(scored), 'group_results': {}}
for group_name, rows in groups.items():
    summary['group_results'][group_name] = {'n': len(rows),
        'baseline_corrected_rigid_mm': float(np.mean([case['baseline_corrected_rigid_mm']
            for case in rows])) if rows else None,
        'baseline_atlas_coverage': float(np.mean([case['baseline_atlas_coverage']
            for case in rows])) if rows else None,
        'measured_oracle_gradient_cosine': (float(np.mean([case[
            'measured_oracle_gradient_cosine'] for case in rows
            if case['measured_oracle_gradient_cosine'] is not None])) if any(case[
            'measured_oracle_gradient_cosine'] is not None for case in rows) else None),
        'directions': {name: {'mean_corrected_rigid_delta_mm': (float(np.mean([case[
            'directions'][name]['corrected_rigid_delta_mm'] for case in rows])) if rows else None),
            'median_corrected_rigid_delta_mm': (float(np.median([case['directions'][name][
            'corrected_rigid_delta_mm'] for case in rows])) if rows else None),
            'fraction_improved_rigid': (float(np.mean([case['directions'][name][
            'corrected_rigid_delta_mm'] < 0 for case in rows])) if rows else None),
            'mean_objective_delta': (float(np.mean([case['directions'][name][
            'objective_delta'] for case in rows])) if rows else None)}
            for name in ('measured_descent', 'random', 'oracle_descent')}}
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
file_sha = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in ('config.json', 'draws.jsonl', 'cases.jsonl', 'summary.json')}
(run / 'completed.json').write_text(json.dumps({'status': 'completed',
    'files_sha256': file_sha, 'source_sha256': source_sha,
    'parent_checkpoint_sha256': parent_sha,
    'attempt_n': len(cases), 'scored_n': len(scored),
    'zero_valid_n': summary['zero_valid_n'], 'nonfinite_n': summary['nonfinite_n'],
    'zero_or_nonfinite_measured_gradient_n': summary['zero_or_nonfinite_measured_gradient_n']},
    indent=2, allow_nan=False))
print(json.dumps({'event': 'TRAIN_direction_complete', 'run': str(run),
    'summary': summary}, allow_nan=False), flush=True)
