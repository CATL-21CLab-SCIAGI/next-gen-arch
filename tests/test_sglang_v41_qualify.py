import copy

import pytest

from archlab.serving.sglang_v41_qualify import score_case


def fixture():
    case = dict(input_ids=[10, 20, 30, 40], target_positions=[1, 3], targets=2)
    response = dict(meta_info=dict(
        input_token_logprobs=[[None, 10], [-.5, 20], [-3., 30], [-1.5, 40]],
        input_top_logprobs=[None, [[-.5, 20], [-1., 21], [-2., 22], [-3., 23], [-4., 24]],
                           [[-1., 21]] * 5,
                           [[-.5, 21], [-1.5, 40], [-2., 22], [-3., 23], [-4., 24]]]))
    return case, response


def test_only_supervised_targets_contribute_with_one_shift():
    case, response = fixture()
    assert score_case(case, response) == dict(targets=2, nll=2., top1=1, top5=2)
    truncated = copy.deepcopy(response)
    for key in ("input_token_logprobs", "input_top_logprobs"):
        truncated["meta_info"][key] = truncated["meta_info"][key][1:]
    assert score_case(case, truncated) == score_case(case, response)


@pytest.mark.parametrize("error", ["alignment", "duplicate", "nonfinite"])
def test_bad_protocol_or_mask_is_rejected(error):
    case, response = fixture()
    if error == "alignment":
        response["meta_info"]["input_token_logprobs"][2][1] = 999
    elif error == "duplicate":
        case["target_positions"] = [1, 1]
    else:
        response["meta_info"]["input_token_logprobs"][1][0] = float("nan")
    with pytest.raises(ValueError):
        score_case(case, response)
