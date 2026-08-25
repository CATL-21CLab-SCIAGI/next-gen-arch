from types import SimpleNamespace

from next_gen_arch.training.megatron_train import _current_training_iteration


def test_current_training_iteration_prefers_live_megatron_counter():
    args = SimpleNamespace(iteration=0, curr_iteration=43)

    assert _current_training_iteration(args) == 43


def test_current_training_iteration_falls_back_before_train_loop():
    args = SimpleNamespace(iteration=7)

    assert _current_training_iteration(args) == 7
