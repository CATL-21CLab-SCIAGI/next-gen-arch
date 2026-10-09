import torch

from archlab.rl.limite_replay import global_signal, trim_replay


def test_trimmed_padding_preserves_causal_loss_gradients_and_global_denominator():
    inputs = dict(
        prompt_ids=torch.tensor([[0, 0, 3, 4]]), prompt_mask=torch.tensor([[0, 0, 1, 1]]),
        completion_ids=torch.tensor([[5, 6, 0, 0]]), completion_mask=torch.tensor([[1, 1, 0, 0]]),
        sampling_per_token_logps=torch.tensor([[-2., -3., 0., 0.]]), advantages=torch.tensor([-1.]),
        num_items_in_batch=torch.tensor(400),
    )
    trimmed = trim_replay(inputs)
    assert trimmed["prompt_ids"].tolist() == [[3, 4]]
    assert trimmed["completion_ids"].tolist() == [[5, 6]]
    assert trimmed["num_items_in_batch"] is inputs["num_items_in_batch"]
    gradients = []
    for batch in (inputs, trimmed):
        parameter = torch.tensor(.05, requires_grad=True)
        ids = torch.cat((batch["prompt_ids"], batch["completion_ids"]), 1).float()
        mask = torch.cat((batch["prompt_mask"], batch["completion_mask"]), 1)
        logps = -(parameter * (ids * mask).cumsum(1)[:, -batch["completion_ids"].shape[1]:])
        ratio = (logps - batch["sampling_per_token_logps"]).exp()
        loss = -(ratio * batch["advantages"][:, None] * batch["completion_mask"]).sum() / batch["num_items_in_batch"]
        loss.backward()
        gradients.append(parameter.grad)
    torch.testing.assert_close(*gradients, rtol=0, atol=0)


def test_signal_only_counts_unmasked_nonzero_advantages():
    inputs = dict(advantages=torch.tensor([[0., 1.]]), completion_mask=torch.tensor([[1, 0]]))
    assert not global_signal(inputs, "cpu")
    inputs["completion_mask"].fill_(1)
    assert global_signal(inputs, "cpu")


def test_all_masked_replay_is_unchanged_for_distributed_dummy_forward():
    batch = dict(prompt_mask=torch.ones(1, 3), completion_mask=torch.zeros(1, 5))
    result = trim_replay(batch)
    assert result["completion_mask"] is batch["completion_mask"]
