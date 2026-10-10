"""Source–atlas evidence for both original and corrected pose branches."""

from training.in_path_global_matcher_128 import global_plane_match_128


def global_plane_evidence_130(model, prediction, mode_index, reflection, offsets,
                              weights, atlas, image_shape, *, candidate_state=None,
                              support_only=False, return_support=False):
    captured = {}

    def save_hidden(module, inputs, output):
        captured['hidden'] = output

    hook = model.global_plane_matcher_120['head'][-2].register_forward_hook(save_hidden)
    try:
        match = global_plane_match_128(model, prediction, mode_index, reflection,
            offsets, weights, atlas, image_shape, side=24,
            candidate_state=candidate_state, support_only=support_only,
            return_support=return_support)
    finally:
        hook.remove()
    evidence = model.global_plane_matcher_120['evidence'](captured['hidden'])
    evidence = evidence.reshape_as(match['input_score'])
    match['evidence_logit'] = evidence
    match['original_score'] = match['input_score'] + evidence
    match['corrected_score'] = match['score'] + evidence
    return match
