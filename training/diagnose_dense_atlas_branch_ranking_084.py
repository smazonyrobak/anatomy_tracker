"""Frozen 083 match-evidence ranking on the blind 061 synthetic DEV beam."""
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
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

pilot = root / 'runs/dense_atlas_correspondence_083_pilot'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
out = root / 'runs/dense_atlas_branch_ranking_084_diagnostic'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / 256 - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


def evidence(logits, support):
    available = support >= .5
    count = available.sum(2)
    peak = logits.masked_fill(~available, -1e4).softmax(2).amax(2)
    chance = count.clamp_min(1).reciprocal()
    excess = torch.where(count >= 2, (peak - chance) / (1 - chance).clamp_min(1e-8), 0.)
    return excess.mean((-2, -1)), (count >= 2).float().mean((-2, -1))


receipt = json.loads((pilot / 'completed.json').read_text())
config = json.loads((pilot / 'config.json').read_text())
parent = Path(config['parent'])
assert receipt['batches'] == 6000 and config['checkpoints'] == [0, 1000, 3000, 6000]
assert not receipt['calibrated'] and not receipt['public_benchmark_used']
assert sha(pilot / 'config.json') == receipt['config_sha256']
assert sha(pilot / 'draws.jsonl') == receipt['draws_sha256']
assert sha(pilot / 'training.jsonl') == receipt['training_sha256']
assert sha(parent) == config['parent_sha256']
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in config['source_sha256'].items())
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in config['synthetic_provenance']['source_sha256'].items())
assert all(sha(pilot / f'match_step_{int(step):05d}.pt') == digest
           for step, digest in receipt['checkpoint_sha256'].items())
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 246 and len({row['synthetic_subject_plan_id'] for row in records}) == 8

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'pilot_completed_sha256': sha(pilot / 'completed.json'),
    'parent_sha256': sha(parent),
    'checkpoint_083_step_6000_sha256': receipt['checkpoint_sha256']['6000'],
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'script_sha256': sha(__file__),
    'beam': 'same blind prior top-8 old and top-6 anchor; no truth insertion',
    'matcher': 'frozen 083 step 6000; no training or score fitting',
    'primary_match_score': 'For each fine/coarse query cell, keep atlas classes with support >= 0.5; p=softmax(masked logits); K=number kept; excess=(max(p)-1/K)/(1-1/K) if K>=2 else 0. Average excess over ALL query cells, including exterior, then average fine/coarse scores. No synthetic valid mask or target enters this score.',
    'truth_role': 'evaluation only: smallest rigid 3D CCF error at 1024 fixed surviving-tissue pixels among existing 14 branches; no supplied correct pose',
    'prior_role': 'frozen parent log_mass plus reflection log probability, comparison only',
    'output_error': 'rigid geometry only, not fitted-map accuracy',
    'candidate_chunk': 2, 'calibrated': False, 'public_benchmark_used': False,
}, indent=2))

frozen = torch.load(parent, map_location='cpu', weights_only=True)
assert frozen['step'] == 50000 and not frozen['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(frozen['model'], strict=True)
model.requires_grad_(False)
del frozen
checkpoint = torch.load(pilot / 'match_step_06000.pt', map_location='cpu', weights_only=True)
assert checkpoint['step'] == 6000 and not checkpoint['calibrated']
head = WholeSliceAtlasFeedback083().cuda().eval()
head.load_state_dict(checkpoint['head'], strict=True)
head.requires_grad_(False)
del checkpoint
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()

cases = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                          prior[:, 32:].topk(6, -1).indices + 32), -1)
        state = prediction['state'].gather(1, (beam // 2)[..., None].expand(-1, -1, 12))
        indices = valid.flatten().nonzero().flatten()
        chosen = indices[torch.linspace(0, len(indices) - 1, 1024,
                                        device='cuda').round().long()]
        chart = torch.stack((chosen.remainder(256),
                             chosen.div(256, rounding_mode='floor')), -1).float()[None] / 256
        truth = target.reshape(-1, 3)[chosen]
        rigid = (points(state, beam % 2, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
        match, support = [], []
        for start in range(0, 14, 2):
            output = head(prediction['feature'], state[:, start:start + 2],
                          beam[:, start:start + 2] % 2, atlas, offsets, weights,
                          match_only=True)
            scores, fractions = [], []
            for scale in ('fine', 'coarse'):
                score, fraction = evidence(output[f'{scale}_match_logits'],
                                           output[f'{scale}_match_support'])
                scores.append(score)
                fractions.append(fraction)
            match.append(torch.stack(scores).mean(0))
            support.append(torch.stack(fractions).mean(0))
        match = torch.cat(match, -1)[0]
        support = torch.cat(support, -1)[0]
        prior_beam = prior.gather(1, beam)[0]
        best = int(rigid.argmin())
        selected = int(match.argmax())
        prior_selected = int(prior_beam.argmax())
        order = match.argsort(descending=True)
        best_rank = int((order == best).nonzero()[0, 0]) + 1
        margin = float(match[best] - match[torch.arange(14, device='cuda') != best].max())
        base = {'section_id': record['section_id'],
                'animal_id': record['animal_id'],
                'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'],
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'panel_physical_section_id': record['panel_physical_section_id'],
                'appearance_mode': record['appearance_mode'],
                'panel_file_sha256': record['sha256']}
        cases.append({**base, 'best_rigid_slot': best,
                      'best_rigid_role': 'old' if int(beam[0, best]) < 32 else 'anchor',
                      'best_rigid_mm': float(rigid[best] / 1000),
                      'match_selected_slot': selected,
                      'match_selected_role': 'old' if int(beam[0, selected]) < 32 else 'anchor',
                      'match_selected_rigid_mm': float(rigid[selected] / 1000),
                      'prior_selected_slot': prior_selected,
                      'prior_selected_rigid_mm': float(rigid[prior_selected] / 1000),
                      'best_rigid_match_rank': best_rank,
                      'best_rigid_match_margin': margin})
        for slot in range(14):
            stream.write(json.dumps({**base, 'beam_slot': slot,
                'branch': int(beam[0, slot]),
                'role': 'old' if int(beam[0, slot]) < 32 else 'anchor',
                'rigid_mm': float(rigid[slot] / 1000),
                'prior_log_score': float(prior_beam[slot]),
                'match_score': float(match[slot]),
                'supported_query_fraction': float(support[slot]),
                'truth_best_existing': slot == best,
                'match_selected': slot == selected,
                'prior_selected': slot == prior_selected}) + '\n')
    stream.flush()


def aggregate(group):
    plans = sorted({row['synthetic_subject_plan_id'] for row in group})
    fields = ('best_rigid_mm', 'match_selected_rigid_mm', 'prior_selected_rigid_mm',
              'best_rigid_match_rank', 'best_rigid_match_margin')
    result = {'sections': len(group), 'plans': len(plans)}
    for field in fields:
        result['plan_equal_' + field] = float(np.mean([
            np.mean([row[field] for row in group if row['synthetic_subject_plan_id'] == plan])
            for plan in plans]))
    for name, limit in (('match_top1', 1), ('match_top3', 3)):
        result['plan_equal_' + name] = float(np.mean([
            np.mean([row['best_rigid_match_rank'] <= limit for row in group
                     if row['synthetic_subject_plan_id'] == plan]) for plan in plans]))
    return result


summary = {'overall': aggregate(cases),
           'by_appearance': {mode: aggregate([row for row in cases if row['appearance_mode'] == mode])
                             for mode in sorted({row['appearance_mode'] for row in cases})},
           'by_best_rigid_role': {role: aggregate([row for row in cases if row['best_rigid_role'] == role])
                                  for role in sorted({row['best_rigid_role'] for row in cases})},
           'by_match_selected_role': {role: aggregate([row for row in cases if row['match_selected_role'] == role])
                                      for role in sorted({row['match_selected_role'] for row in cases})},
           'scope': 'Frozen synthetic geometry diagnostic only; no fitted-map, real-animal, calibration or public-benchmark claim. Failure of this fixed evidence score does not disprove a trainable ranker.'}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'sections': len(cases), 'rows': 14 * len(cases),
    'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2))
