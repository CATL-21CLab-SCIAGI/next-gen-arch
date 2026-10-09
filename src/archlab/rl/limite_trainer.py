"""Upstream GRPO with recorded sampling probabilities and bounded replay."""

import time
from types import MethodType

import torch
from trl import GRPOTrainer

from archlab.rl.grpo_log_collection import bind_host_object_gather
from archlab.rl.limite_replay import global_signal, trim_replay
from archlab.rl.limite_rollout import with_behavior_logprobs


class BehaviorGRPO(GRPOTrainer):
    def _generate_and_score_completions(self, *args, **kwargs):
        upstream = super()._generate_and_score_completions
        rendezvous = getattr(self, "archlab_rollout_rendezvous", None)
        if rendezvous is None:
            return upstream(*args, **kwargs)
        bound = getattr(self, "_archlab_host_generation", None)
        if bound is None:
            bound = MethodType(
                bind_host_object_gather(upstream.__func__, rendezvous.gather_object), self
            )
            self._archlab_host_generation = bound
        return bound(*args, **kwargs)

    def _prepare_inputs(self, inputs):
        prepared = getattr(self, "_archlab_prepared_once", None)
        if prepared is not None:
            self._archlab_prepared_once = None
            return prepared
        result = super()._prepare_inputs(inputs)
        return trim_replay(result) if getattr(self, "archlab_trim_replay", False) else result

    def training_step(self, model, inputs, num_items_in_batch):
        if not getattr(self, "archlab_skip_flat_backward", False):
            return super().training_step(model, inputs, num_items_in_batch)
        if self.beta != 0:
            raise ValueError("flat-advantage skipping requires the qualified zero-KL objective")
        # Trainer normally restores train mode before preparing inputs. Our
        # early signal check must do the same after evaluation, or GRPO returns
        # an unsplit evaluation batch and leaves its training replay stale.
        model.train()
        started = time.perf_counter()
        prepared = self._prepare_inputs(inputs)
        if global_signal(prepared, self.accelerator.device):
            self._archlab_prepared_once = prepared
            prepared_seconds = time.perf_counter() - started
            output = super().training_step(model, inputs, num_items_in_batch)
            # The upstream timer starts after our preparation, which includes
            # generation. Restore that duration in the same step metric.
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics["train"]["step_time"][-1] += prepared_seconds
            else:
                self._current_train_step_time += prepared_seconds
            return output
        self._step += 1
        self._current_train_step_time += time.perf_counter() - started
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        self._metrics["train"]["replay/skipped_global_no_signal"].append(1.0)
        return torch.zeros((), device=self.accelerator.device)

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **kwargs
    ):
        if getattr(self, "archlab_chunked_policy_scores", False):
            from archlab.rl.limite_scoring import native_replay_scores

            compute_entropy = kwargs.pop("compute_entropy", False)
            if any(value is not None for value in kwargs.values()):
                raise ValueError("native chunked replay only supports text inputs")
            return native_replay_scores(model, input_ids, attention_mask, logits_to_keep,
                                        temperature=self.temperature, compute_entropy=compute_entropy)
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, batch_size=1, **kwargs
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(
            model, with_behavior_logprobs(inputs), return_outputs, num_items_in_batch
        )
        loss = result[0] if isinstance(result, tuple) else result
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("nonfinite GRPO loss; refusing backward and optimizer update")
        return result
