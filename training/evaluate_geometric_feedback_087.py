"""Matched synthetic readout of correspondence-derived plane updates on the frozen 085 beam."""
import hashlib
import json
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.whole_slice_atlas_geometric_feedback_087 import WholeSliceAtlasGeometricFeedback087

panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
run = root / 'runs/allbeam_fitted_ranker_085_pilot'
baseline_run = root / 'runs/allbeam_fitted_ranker_085_development_eval'
rank_run = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
out = root / 'runs/geometric_feedback_087_development_eval'
side = 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


def fit_summary(mapped):
    local = mapped['local_displacement_um'] / 1000
    magnitude = local.square().sum(2).sqrt().mean((-2, -1))
    roughness = ((local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
                 + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1)))
    support = mapped['atlas_pair'][:, :, 1].mean((-2, -1))
    reliability = mapped['correspondence_logit'].sigmoid().mean((-3, -2, -1))
    return torch.stack((magnitude, roughness, support, reliability), -1)


def candidate_prediction(prediction, choice, state):
    mode = choice // 2
    return {**prediction, 'state': state,
            'log_mass': prediction['log_mass'].gather(1, mode),
            'reflection_logit': prediction['reflection_logit'].gather(1, mode)}


def field_at_chart(surface, chart):
    count = surface.shape[1]
    grid = (chart + .5 / side) * 2 - 1
    grid = grid[None].expand(count, -1, -1).reshape(count, 1, len(chart), 2)
    return F.grid_sample(surface[0].permute(0, 3, 1, 2), grid,
                         padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)


training = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
old_receipt = json.loads((baseline_run / 'completed.json').read_text())
rank_receipt = json.loads((rank_run / 'completed.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert finished['batches'] == training['batches'] == 1000
assert not finished['calibrated'] and not finished['public_benchmark_used']
assert sha(run / 'config.json') == finished['config_sha256']
assert sha(run / 'draws.jsonl') == finished['draws_sha256']
assert sha(run / 'training.jsonl') == finished['training_sha256']
assert sha(run / 'ranker_step_01000.pt') == finished['checkpoint_sha256']['1000']
assert sha(training['parent']) == training['parent_sha256']
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in training['source_sha256'].items())
assert sha(baseline_run / 'config.json') == old_receipt['config_sha256']
assert sha(baseline_run / 'rows.jsonl') == old_receipt['rows_sha256']
assert sha(baseline_run / 'summary.json') == old_receipt['summary_sha256']
assert sha(rank_run / 'config.json') == rank_receipt['config_sha256']
assert sha(rank_run / 'candidates.jsonl') == rank_receipt['candidates_sha256']
assert sha(rank_run / 'sections.jsonl') == rank_receipt['sections_sha256']
assert sha(rank_run / 'summary.json') == rank_receipt['summary_sha256']
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
records = [r for r in map(json.loads, (panel / 'records.jsonl').open()) if r['eligible']]
baseline = {r['section_id']: r for r in map(json.loads, (baseline_run / 'rows.jsonl').open())
            if r['set'] == 'synthetic' and r['step'] == 1000}
ranked = {}
for candidate in map(json.loads, (rank_run / 'candidates.jsonl').open()):
    ranked.setdefault(candidate['section_id'], []).append(candidate['branch_id'])
assert len(records) == len(baseline) == len(ranked) == 246
assert len({r['synthetic_subject_plan_id'] for r in records}) == 8

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
head = WholeSliceAtlasGeometricFeedback087().cuda().eval()
checkpoint = torch.load(run / 'ranker_step_01000.pt', map_location='cpu', weights_only=True)
assert checkpoint['step'] == 1000 and not checkpoint['calibrated']
model.load_state_dict(checkpoint['model'], strict=True)
head.load_state_dict(checkpoint['feedback'], strict=True)
model.requires_grad_(False)
head.requires_grad_(False)
del checkpoint

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'checkpoint_sha256': finished['checkpoint_sha256']['1000'],
    'training_receipt_sha256': sha(run / 'completed.json'),
    'baseline_receipt_sha256': sha(baseline_run / 'completed.json'),
    'ranking_receipt_sha256': sha(rank_run / 'completed.json'),
    'panel_receipt_sha256': sha(panel / 'completed.json'),
    'protocol_sha256': sha(Path(__file__).parent.parent / 'docs/publication'
                           / 'GEOMETRIC_ATLAS_FEEDBACK_087_PROTOCOL_20261004.md'),
    'source_sha256': {n: sha(Path(__file__).parent / n) for n in (
        'evaluate_geometric_feedback_087.py',
        'whole_slice_atlas_geometric_feedback_087.py')},
    'beam': 'identical 8 old + 6 anchor blind branches from frozen 085',
    'change': '083 posterior moments give a damped local rigid 6-DOF update; all weights remain frozen',
    'metric': 'mean 3-D CCF error on surviving tissue; mapped96 uses fixed 1024 pixels; mapped256 uses all',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        indices = valid.flatten().nonzero().flatten()
        chosen = indices[torch.linspace(0, len(indices) - 1, 1024,
                                        device='cuda').round().long()]
        chart = torch.stack((chosen.remainder(side),
            chosen.div(side, rounding_mode='floor')), -1).float() / side
        reference = target.reshape(-1, 3)[chosen]
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)
        assert choice[0].tolist() == ranked[record['section_id']]
        state = prediction['state'].gather(
            1, (choice // 2)[..., None].expand(-1, -1, 12))
        states, scores, surfaces, evidence = [], [], [], []
        for start in range(0, 14, 2):
            branch = choice[:, start:start + 2]
            reflection = branch % 2
            first = head(prediction['feature'], state[:, start:start + 2],
                         reflection, atlas, offsets, weights)
            selected = candidate_prediction(prediction, branch, first['state'])
            index = torch.arange(2, device='cuda')[None]
            mapped64 = model.map(selected, offsets, index, reflection, (64, 64),
                                 atlas, weights, return_refinement_feature=True,
                                 feature_side=64, source_shape=(side, side),
                                 spatial_evidence=first['spatial_evidence'])
            second = head(prediction['feature'], first['state'], reflection,
                          atlas, offsets, weights, fit_summary=fit_summary(mapped64))
            selected['state'] = second['state']
            mapped96 = model.map(selected, offsets, index, reflection, (96, 96),
                                 atlas, weights, feature_side=96,
                                 source_shape=(side, side),
                                 spatial_evidence=second['spatial_evidence'])
            score = model.score_fitted_candidates(image, selected, mapped96,
                                                  atlas, weights) + second['quality_logit']
            states.append(second['state'])
            scores.append(score)
            surfaces.append(mapped96['centre_surface_ccf_ap_dv_ml_um'])
            evidence.append(second['spatial_evidence'])
        state = torch.cat(states, 1)
        score = torch.cat(scores, 1)
        surface = torch.cat(surfaces, 1)
        evidence = torch.cat(evidence, 1)
        reflection = choice % 2
        rigid = (points(state, reflection, chart) - reference).norm(dim=-1).mean(-1)[0]
        mapped96 = (field_at_chart(surface, chart) - reference[None]).norm(dim=-1).mean(-1)
        selected_index = int(score[0].argmax())
        selected_choice = choice[:, selected_index:selected_index + 1]
        selected = candidate_prediction(prediction, selected_choice,
                                        state[:, selected_index:selected_index + 1])
        mapped256 = model.map(selected, offsets, torch.zeros((1, 1), device='cuda',
            dtype=torch.long), reflection[:, selected_index:selected_index + 1],
            (side, side), atlas, weights, source_shape=(side, side),
            spatial_evidence=evidence[:, selected_index:selected_index + 1])
        field = mapped256['centre_surface_ccf_ap_dv_ml_um'][0, 0]
        final_error = (field[valid] - target[valid]).norm(dim=-1).mean()
        old = baseline[record['section_id']]
        assert old['sha256'] == record['sha256']
        row = {k: record[k] for k in ('section_id', 'animal_id', 'synthetic_animal_id',
            'specimen_id', 'experiment_id', 'synthetic_subject_plan_id',
            'panel_physical_section_id', 'sha256', 'appearance_mode')}
        row.update({'valid_pixels': record['valid_pixels'],
                    'branches': choice[0].tolist(),
                    'scores': score[0].tolist(),
                    'rigid_errors_um': rigid.tolist(),
                    'mapped96_errors_um': mapped96.tolist(),
                    'selected_branch': int(selected_choice[0, 0]),
                    'selected_mapped96_um': float(mapped96[selected_index]),
                    'best14_mapped96_um': float(mapped96.min()),
                    'selected_mapped256_um': float(final_error),
                    'baseline_selected_branch': old['model_selected_branch'],
                    'baseline_selected_mapped96_um': old['model_selected_mapped96_um'],
                    'baseline_best14_mapped96_um': old['model_best14_mapped96_um'],
                    'baseline_selected_mapped256_um': old['model_selected_mapped256_um'],
                    'parent_selected_mapped256_um': old['parent_selected_mapped256_um']})
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0 or len(rows) == len(records):
            stream.flush()
            print(json.dumps({'sections': len(rows), 'total': len(records)}), flush=True)

plans = sorted({r['synthetic_subject_plan_id'] for r in rows})
metrics = ('selected_mapped96_um', 'best14_mapped96_um',
           'selected_mapped256_um', 'baseline_selected_mapped96_um',
           'baseline_best14_mapped96_um', 'baseline_selected_mapped256_um',
           'parent_selected_mapped256_um')
by_plan = {plan: {'sections': sum(r['synthetic_subject_plan_id'] == plan for r in rows),
    **{m: float(np.mean([r[m] for r in rows if r['synthetic_subject_plan_id'] == plan]))
       for m in metrics}} for plan in plans}
summary = {'sections': len(rows), 'plans': len(plans), 'by_plan': by_plan,
    'plan_equal_mean_um': {m: float(np.mean([by_plan[p][m] for p in plans]))
                           for m in metrics},
    'selected_branch_changed': sum(r['selected_branch'] != r['baseline_selected_branch']
                                   for r in rows)}
summary['by_appearance'] = {}
for appearance in sorted({r['appearance_mode'] for r in rows}):
    group = [r for r in rows if r['appearance_mode'] == appearance]
    represented = sorted({r['synthetic_subject_plan_id'] for r in group})
    summary['by_appearance'][appearance] = {
        'sections': len(group), 'plans': len(represented),
        'plan_equal_mean_um': {m: float(np.mean([
            np.mean([r[m] for r in group if r['synthetic_subject_plan_id'] == plan])
            for plan in represented])) for m in metrics}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
