"""One explicit experimental whole-model inference, used by the GUI executor."""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from joint_slice_input import prepare_joint_slice_input


def run_joint_slice(checkpoint_path, image, raw_to_oriented, raw_shape, brush_mask,
                    source_sha256, image_sha256, atlas_volume, annotation_volume,
                    atlas_hashes, thickness_um, device, messages, cancel_event, version='v7'):
    root = Path('I:/AnatomyTracker')
    for variable, directory in (('TEMP', 'tmp'), ('TMP', 'tmp'),
                                ('TORCH_HOME', 'cache/torch'), ('CUDA_CACHE_PATH', 'cache/cuda')):
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    sys.dont_write_bytecode = True
    repository = Path(__file__).resolve().parents[1]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    import torch
    from training.arbitrary_plane_allen_atlas_binding_v6 import (
        TEMPLATE_RAW_SHA256_V6, ANNOTATION_RAW_SHA256_V6, ATLAS_SHAPE_AP_DV_ML_V6,
    )
    if version == 'v7':
        from training.arbitrary_plane_joint_inference_v7 import load_joint_v7_checkpoint as load_checkpoint
        from training.arbitrary_plane_joint_inference_v7 import infer_joint_v7 as infer_joint
    elif version == 'v8':
        from training.arbitrary_plane_joint_inference_v8 import load_joint_v8_checkpoint as load_checkpoint
        from training.arbitrary_plane_joint_inference_v8 import infer_joint_v8 as infer_joint
    else:
        raise ValueError(f'Unsupported joint-model version: {version}')

    if cancel_event.is_set():
        raise InterruptedError
    path = Path(checkpoint_path).resolve()
    if path.drive.upper() != 'I:':
        raise ValueError('Joint-model development checkpoints must be on I:')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select CPU explicitly to use CPU inference.')
    if (atlas_hashes != {'average_template_25.nrrd': TEMPLATE_RAW_SHA256_V6,
                         'annotation_25.nrrd': ANNOTATION_RAW_SHA256_V6}
            or tuple(atlas_volume.shape) != ATLAS_SHAPE_AP_DV_ML_V6):
        raise ValueError('This joint checkpoint requires its pinned Allen CCFv3 2017 25 um atlas.')
    messages.put((0, 'Loading the selected whole checkpoint; no legacy model is used.'))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    model, config = load_checkpoint(path, device=device)
    if f'arbitrary_plane_joint_model_{version}.py' not in config['source_sha256']:
        raise ValueError(f'Selected checkpoint is not a whole-model {version} checkpoint')
    if version == 'v8' and model.modes != 8:
        raise ValueError('Joint v8 GUI inference requires eight modes (16 branches).')
    for name, expected in config['source_sha256'].items():
        if name in ('train_joint_v7.py', 'train_joint_v7_expanded.py',
                    'train_joint_v7_streaming.py', 'train_joint_v8_feedback_pilot.py',
                    'train_joint_v8_million_direct.py'):
            continue  # Training schedules do not change inference semantics.
        current = repository / 'training' / name
        if hashlib.sha256(current.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Checkpoint implementation differs from installed source: {name}')
    if cancel_event.is_set():
        raise InterruptedError
    prepared = prepare_joint_slice_input(image, raw_to_oriented, raw_shape,
                                         tuple(config['resolution']), brush_mask=brush_mask)
    prepared.update(source_sha256=source_sha256, raw_to_oriented_xy=raw_to_oriented,
                    oriented_image_sha256=image_sha256)
    # Exactly the fixed training-atlas intensity transform, not display contrast.
    intensity = np.clip((atlas_volume.astype(np.float32) - np.float32(9)) / np.float32(264), 0, 1)
    intensity[annotation_volume == 0] = 0
    atlas = torch.from_numpy(intensity[None]).to(device)
    inputs = torch.from_numpy(prepared['channels'][None]).to(device)
    offsets = torch.linspace(-thickness_um / 2, thickness_um / 2, 9, device=device)[None]
    weights = torch.ones_like(offsets)
    weights[:, [0, -1]] = .5
    weights /= weights.sum(-1, keepdim=True)
    if cancel_event.is_set():
        raise InterruptedError
    messages.put((0, 'Fitting all predicted locations, two at a time. Cancel discards the result when inference returns.'))
    prediction = infer_joint(model, inputs, atlas, offsets, weights, context=None, chunk=2)
    if cancel_event.is_set():
        raise InterruptedError
    prediction['runtime'] = {'device': device, 'section_thickness_um': float(thickness_um),
                             'psf': 'assumed uniform through-plane profile; nine-node trapezoidal integration',
                             'checkpoint_path': str(path), 'training_scope': config['scope'],
                             'model_version': version}
    return prepared, prediction, digest
