"""Matched fitter-only continuation: image-only versus one-pass atlas-conditioned warp."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

PARENT_RUN = ROOT / 'runs/one_shot_joint_mixed_real_003'
PARENT = PARENT_RUN / 'joint_step_20000.pt'
DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
RUN = ROOT / 'runs/one_shot_atlas_conditioned_warp_matched_001'
SEED, UPDATES, BATCH, SIDE = 2026100401, 4000, 3, 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert json.loads((PARENT_RUN / 'completed.json').read_text())['updates'] == 20000
parent = torch.load(PARENT, map_location='cpu', weights_only=True)
assert parent['step'] == 20000 and not parent['calibrated']
synthetic = load_streaming_synthetic_v7_64(device='cuda')
records = [json.loads(line) for line in (DEV / 'records.jsonl').read_text().splitlines()]
assert len(records) == 64 and len({row['animal_id'] for row in records}) == 4
assert all(row['split'] == 'development' for row in records)
assert not {row['animal_id'] for row in records} & {
    base['lineage']['animal_id'] for base in synthetic['bases']}
RUN.mkdir(parents=True, exist_ok=False)
config = {'parent': str(PARENT), 'parent_sha256': hashlib.sha256(PARENT.read_bytes()).hexdigest(),
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
              for name in ('arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_stream.py',
                           'arbitrary_plane_streaming_synthetic_v7_64.py', Path(__file__).name)},
          'synthetic_provenance': synthetic['provenance'], 'dev_records_sha256': hashlib.sha256(
              (DEV / 'records.jsonl').read_bytes()).hexdigest(),
          'seed': SEED, 'updates_per_arm': UPDATES, 'batch': BATCH, 'image_side': SIDE,
          'arms': ['image_only', 'atlas_conditioned'],
          'trainable': 'existing warp_shared/warp_condition/warp; conditioned arm additionally trains new atlas_encoder/pair; pose and common image encoder frozen',
          'optimizer': 'fresh AdamW for trainable fitter weights in both arms; not parent optimizer continuation',
          'conditioning': 'one finite-thickness zero-warp atlas render at supplied pose/reflection, normalized atlas intensity+support, feature difference and local 5x5 correlation; no detach from pose',
          'development': 'one preselected appearance per held-out physical section; native192 images and CCF targets resized to256 for comparison with previous readout; ineligible rows retained but not scored',
          'scope': 'internal fitter ablation only; real weak labels, public benchmark, calibration and final-test animals unused'}
(RUN / 'config.json').write_text(json.dumps(config, indent=2))


def development(arm, step, model):
    model.eval()
    rows = []
    with torch.inference_mode():
        for record in records:
            appearance = record['section_index'] % 3
            with np.load(DEV / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][appearance:appearance + 1].copy()).cuda()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                valid = torch.from_numpy((arrays['visible_support'][appearance:appearance + 1] > .25).copy()).cuda()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                reflection = torch.tensor([[int(arrays['reflection'])]], device='cuda')
                eligible = bool(arrays['eligible'][appearance])
            image = F.interpolate(image, (SIDE, SIDE), mode='bilinear', align_corners=False)
            target = F.interpolate(target.permute(0, 3, 1, 2), (SIDE, SIDE), mode='bilinear',
                                   align_corners=False).permute(0, 2, 3, 1)
            valid = F.interpolate(valid[:, None].float(), (SIDE, SIDE), mode='nearest')[:, 0] > .5
            prediction = model.predict(image)
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                          F.logsigmoid(prediction['reflection_logit'])), -1)
            selected = int((prediction['log_mass'][..., None] + log_reflection).flatten(1).argmax(-1)[0])
            row = {'arm': arm, 'step': step, 'appearance': record['modes'][appearance],
                   'eligible': eligible, 'valid_pixels': int(valid.sum()),
                   'selected_mode': selected // 2, 'selected_reflection': selected % 2,
                   'true_reflection': int(reflection[0, 0]),
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')}}
            for label in ('true', 'predicted'):
                mode = torch.tensor([[0 if label == 'true' else selected // 2]], device='cuda')
                flag = reflection if label == 'true' else torch.tensor([[selected % 2]], device='cuda')
                state = prediction['state'].clone()
                if label == 'true':
                    state[:, 0] = truth
                mapped = model.map({**prediction, 'state': state}, offsets, mode, flag,
                                   (SIDE, SIDE), synthetic['atlas'], weights)
                surface = mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0]
                field = mapped['local_displacement_um'][:, 0]
                frame = full_frame_state_to_components(mapped['state'][:, 0])[1]
                rigid = surface - torch.einsum('bij,bjhw->bhwi', frame, field)
                if bool(valid.any()):
                    zero = float((rigid - target).norm(dim=-1)[valid].mean())
                    warped = float((surface - target).norm(dim=-1)[valid].mean())
                    row.update({f'{label}_zero_warp_um': zero, f'{label}_warped_um': warped,
                                f'{label}_warp_gain_um': zero - warped})
                else:
                    row.update({f'{label}_zero_warp_um': None, f'{label}_warped_um': None,
                                f'{label}_warp_gain_um': None})
            rows.append(row)
    with (RUN / arm / 'development.jsonl').open('a') as output:
        output.writelines(json.dumps(row) + '\n' for row in rows)
    metrics = ('true_zero_warp_um', 'true_warped_um', 'true_warp_gain_um',
               'predicted_zero_warp_um', 'predicted_warped_um', 'predicted_warp_gain_um')
    summary = {'arm': arm, 'step': step}
    for stratum in ('eligible', 'all', 'censored'):
        subset = [row for row in rows if (stratum == 'all' or row['eligible'] == (stratum == 'eligible'))
                  and row['true_warped_um'] is not None]
        donors = sorted({row['animal_id'] for row in subset})
        summary[stratum] = {'sections': len(subset), 'animals': len(donors),
            **{metric: float(np.mean([np.mean([row[metric] for row in subset if row['animal_id'] == donor])
                                       for donor in donors])) if donors else None for metric in metrics}}
    with (RUN / arm / 'development_summary.jsonl').open('a') as output:
        output.write(json.dumps(summary) + '\n')
    model.train()
    print(json.dumps({'arm': arm, 'event': 'development', 'step': step,
                      'eligible': summary['eligible']}), flush=True)


matched = []
for arm in ('image_only', 'atlas_conditioned'):
    directory = RUN / arm
    directory.mkdir()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = OneShotJointSliceModel(atlas_conditioning=(arm == 'atlas_conditioned')).cuda()
    missing, unexpected = model.load_state_dict(parent['model'], strict=(arm == 'image_only'))
    assert not unexpected and (not missing if arm == 'image_only' else
                               set(missing) == {name for name in model.state_dict()
                                                if name.startswith(('atlas_encoder.', 'pair.'))})
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [model.warp_shared, model.warp_condition, model.warp]
    if arm == 'atlas_conditioned':
        modules.extend((model.atlas_encoder, model.pair))
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=3e-5, weight_decay=1e-4)
    subjects_rng = torch.Generator().manual_seed(SEED)
    draw_seed = SEED * 10000000
    started = time.perf_counter()
    development(arm, 0, model)
    with (directory / 'training.jsonl').open('w') as log, (RUN / 'shared_draws.jsonl').open(
            'w' if arm == 'image_only' else 'a') as draws:
        for step in range(1, UPDATES + 1):
            accepted, pending, signature = {}, list(range(BATCH)), []
            while pending:
                virtual = torch.randint(len(synthetic['subjects']), (len(pending),), generator=subjects_rng).tolist()
                sampled = sample_one_shot_stream(synthetic, virtual, draw_seed, side=SIDE)
                draw_seed += 1
                for index, identity in enumerate(sampled['provenance']):
                    slot = pending[index]
                    used = bool(sampled['eligible'][index])
                    signature.append((slot, identity, used))
                    if arm == 'image_only':
                        draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                    if used:
                        accepted[slot] = {key: sampled[key][index:index + 1] for key in
                                          ('inputs', 'state', 'reflection', 'offsets', 'weights', 'centre', 'valid_mask')}
                pending = [slot for slot in pending if slot not in accepted]
            if arm == 'image_only':
                matched.append(signature)
            else:
                assert signature == matched[step - 1]
            batch = {key: torch.cat([accepted[slot][key] for slot in range(BATCH)]) for key in accepted[0]}
            prediction = model.predict(batch['inputs'])
            state = prediction['state'].clone()
            state[:, 0] = batch['state']
            mapped = model.map({**prediction, 'state': state}, batch['offsets'],
                               torch.zeros((BATCH, 1), dtype=torch.long, device='cuda'),
                               batch['reflection'][:, None], (SIDE, SIDE), synthetic['atlas'], batch['weights'])
            valid = batch['valid_mask'].float()
            error = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - batch['centre']).norm(dim=-1)
            mapping = (error * valid).sum() / valid.sum() / 500
            field = mapped['local_displacement_um'][:, 0]
            magnitude = field.square().mean().sqrt() / 1000
            smooth = ((field[:, :, 1:] - field[:, :, :-1]).abs().mean()
                      + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()) / 200
            loss = mapping + .1 * magnitude + .02 * smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad), 5.,
                error_if_nonfinite=True)
            optimizer.step()
            row = {'arm': arm, 'step': step, 'presentations': step * BATCH,
                   'mapping_um': float(mapping.detach() * 500), 'field_rms_um': float(magnitude.detach() * 1000),
                   'smooth': float(smooth.detach()), 'loss': float(loss.detach()),
                   'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 200 == 0:
                log.flush()
                if arm == 'image_only':
                    draws.flush()
                print(json.dumps(row), flush=True)
            if step % 2000 == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'step': step, 'arm': arm, 'config': config, 'calibrated': False},
                           directory / f'warp_step_{step:05d}.pt')
                development(arm, step, model)
    (directory / 'completed.json').write_text(json.dumps({'arm': arm, 'updates': UPDATES,
        'presentations': UPDATES * BATCH, 'seconds': time.perf_counter() - started,
        'public_benchmark_used': False, 'calibrated': False}, indent=2))
    del model, optimizer
print('Matched one-shot warp arms complete; frozen readout audit required', flush=True)
