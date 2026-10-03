"""Frozen test of whether patch evidence can correct direct pose proposals."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state, full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025
from training.atlas_pose_patch_fit_035 import patch_fit

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/atlas_pose_patch_fit_035'
pose_path = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
patch_path = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
chosen = []
for subject in sorted({row['synthetic_subject_plan_id'] for row in records}):
    chosen.extend(sorted((row for row in records if row['eligible'] and
                          row['synthetic_subject_plan_id'] == subject), key=lambda row: row['sha256'])[:3])
assert len(chosen) == 24
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
pose = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
pose.load_state_dict(torch.load(pose_path, map_location='cpu', weights_only=True)['model'])
patch = AtlasOrientedPatch025().cuda().eval()
patch.load_state_dict(torch.load(patch_path, map_location='cpu', weights_only=True)['model'])
patch.requires_grad_(False)
pixel_yx = torch.tensor([(y, x) for y in (48, 128, 208) for x in (48, 128, 208)],
                        device='cuda', dtype=torch.float32)
out.mkdir(parents=True, exist_ok=True)
config = {'pose_checkpoint_sha256': sha(pose_path), 'patch_checkpoint_sha256': sha(patch_path),
          'panel_records_sha256': sha(panel / 'records.jsonl'), 'sections': 24,
          'selection': 'SHA-sorted first three eligible cases per each of eight synthetic DEV identities',
          'candidate_pool': '019 direct prior top eight pose/reflection branches plus truth diagnostic only',
          'query_pixels_yx': pixel_yx.int().cpu().tolist(), 'patch_side': 64,
          'score': 'mean diagonal cosine of frozen 025 observed/finite-thickness-atlas descriptors',
          'gradient_probe': 'at prior top1, oracle best8, and truth perturbed by 0.1 rad/1 mm; cosine between descriptor-ascent and physical-error-descent gradients in bounded local pose coordinates, plus one 0.02-norm gradient step',
          'development_gate': 'mean descriptor-selected rigid error at least 0.25 mm below 019 prior top1 AND exact plane score above candidate median in at least 70 percent of sections AND prior-top1 descriptor gradient improves physical error in at least 60 percent of sections',
          'no_training_or_gt_selection': True, 'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('diagnose_atlas_pose_patch_fit_035.py', 'atlas_pose_patch_fit_035.py')}}
(out / 'config.json').write_text(json.dumps(config, indent=2))


def rigid_error(state, reflection, target, valid):
    centre, frame, basis = full_frame_state_to_components(state)
    ids = valid.flatten().nonzero().flatten()
    chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')), -1).float() / side
    chart = chart[None].expand(len(state), -1, -1).clone()
    chart[..., 0] = torch.where(reflection[:, None].bool(), (side - 1) / side - chart[..., 0], chart[..., 0])
    estimate = centre[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, chart - .5)
    return (estimate - target.reshape(-1, 3)[ids][None]).norm(dim=-1).mean(-1)


rows = []
limits = torch.tensor((.9, .9, .9, 4000., 4000., 4000., .25, .25, .2), device='cuda')
with (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(chosen, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            truth_reflection = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        with torch.no_grad():
            prediction = pose.predict(image)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            choice = prior.topk(8).indices
            states = prediction['state'][0, choice // 2]
            reflections = choice % 2
            all_states = torch.cat((states, truth), 0)
            all_reflections = torch.cat((reflections, reflections.new_tensor([truth_reflection])), 0)
            scores = torch.cat([patch_fit(patch, image, atlas, all_states[first:first + 2],
                             all_reflections[first:first + 2], offsets.expand(min(2, 9 - first), -1),
                             weights.expand(min(2, 9 - first), -1), pixel_yx)[0]
                             for first in range(0, 9, 2)])
            error = rigid_error(all_states, all_reflections, target, valid)
            perturbed = compose_full_frame_state(truth, truth.new_tensor(
                [[.10, -.06, .03, 1000., -500., 250., 0., 0., 0.]]))
        probe = {}
        for name, base, reflection in (
                ('prior_top1', states[:1], reflections[:1]),
                ('oracle8', states[error[:8].argmin():error[:8].argmin() + 1],
                 reflections[error[:8].argmin():error[:8].argmin() + 1]),
                ('perturbed_truth', perturbed, all_reflections[8:9])):
            q = torch.zeros((1, 9), device='cuda', requires_grad=True)
            moved = compose_full_frame_state(base, q * limits)
            score = patch_fit(patch, image, atlas, moved, reflection,
                              offsets, weights, pixel_yx)[0].sum()
            gradient_score = torch.autograd.grad(score, q, retain_graph=True)[0]
            physical = rigid_error(moved, reflection, target, valid).sum()
            gradient_physical = torch.autograd.grad(physical, q)[0]
            direction = gradient_score / gradient_score.norm().clamp_min(1e-12)
            cosine = F.cosine_similarity(gradient_score, -gradient_physical, dim=-1)
            with torch.no_grad():
                moved_after = compose_full_frame_state(base, .02 * direction * limits)
                after = rigid_error(moved_after, reflection, target, valid)
            probe[name] = {'gradient_cosine': float(cosine),
                           'before_um': float(physical.detach()), 'after_um': float(after)}
        selected = int(scores[:8].argmax())
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                   'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row.update(prior_top1_um=float(error[0]), selected_um=float(error[selected]),
                   oracle8_um=float(error[:8].min()), truth_um=float(error[8]),
                   selected_branch=int(choice[selected]), candidate_branches=choice.cpu().tolist(),
                   candidate_scores=scores[:8].cpu().tolist(), exact_pose_score=float(scores[8]),
                   median_candidate_score=float(scores[:8].median()), gradient_probe=probe)
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 8 == 0:
            stream.flush()
            print(json.dumps({'sections': number, 'of': 24}), flush=True)

prior = np.asarray([row['prior_top1_um'] for row in rows])
selected = np.asarray([row['selected_um'] for row in rows])
oracle = np.asarray([row['oracle8_um'] for row in rows])
exact_above = np.asarray([row['exact_pose_score'] > row['median_candidate_score'] for row in rows])
gradient_better = np.asarray([row['gradient_probe']['prior_top1']['after_um'] <
                              row['gradient_probe']['prior_top1']['before_um'] for row in rows])
result = {'sections': len(rows), 'synthetic_identities': 8,
          'mean_prior_top1_um': float(prior.mean()), 'mean_selected_um': float(selected.mean()),
          'mean_oracle8_um': float(oracle.mean()), 'exact_score_above_median_fraction': float(exact_above.mean()),
          'selected_better_than_prior_fraction': float((selected < prior).mean()),
          'prior_top1_gradient_positive_fraction': float(gradient_better.mean()),
          'gradient_cosine_mean': {name: float(np.mean([
              row['gradient_probe'][name]['gradient_cosine'] for row in rows]))
              for name in ('prior_top1', 'oracle8', 'perturbed_truth')},
          'development_gate': bool(selected.mean() <= prior.mean() - 250 and
                                   exact_above.mean() >= .7 and gradient_better.mean() >= .6),
          'scope': 'frozen synthetic DEV diagnostic only; no direct-pose training, uncertainty calibration, real-animal validation or benchmark'}
(out / 'summary.json').write_text(json.dumps(result, indent=2))
(out / 'completed.json').write_text(json.dumps({name + '_sha256': sha(out / f'{name}.json') for name in
    ('config', 'summary')} | {'rows_sha256': sha(out / 'rows.jsonl')}, indent=2))
print(json.dumps(result), flush=True)
