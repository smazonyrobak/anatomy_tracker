"""Fixed image-only photometric structure channels for the 137 comparison."""

import torch
import torch.nn.functional as F


def fill_structure(inputs):
    image = inputs[:, :1]
    scale = torch.quantile(image.flatten(1), .99, dim=1).clamp_min(1e-4)[:, None, None, None]
    global_contrast = (image / scale).clamp(0, 2)
    mean = F.avg_pool2d(global_contrast, 17, 1, 8, count_include_pad=False)
    variance = F.avg_pool2d(global_contrast.square(), 17, 1, 8,
                            count_include_pad=False) - mean.square()
    local_contrast = ((global_contrast - mean) / (variance.clamp_min(0).sqrt() + .05)).clamp(-3, 3)
    output = inputs.clone()
    output[:, 3:4] = global_contrast
    output[:, 4:5] = local_contrast
    return output
