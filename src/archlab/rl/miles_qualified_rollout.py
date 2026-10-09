"""Native Miles rollout with a verified first batch and bounded partial age.

Generation, filtering, partial continuation, evaluation and optimization remain
upstream. The optional bootstrap consumes an actual unupdated-parent rollout
once, allowing the full training path to run before another expensive rollout.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn

from archlab.artifacts import sha256_file
from archlab.rl.miles_rollout_metrics import generated_tokens_at_version


def split_stale_groups(groups, current_version, max_age):
    """Age is in published policy versions, not collection time or HTTP age."""
    retained, stale = [], []
    for group in groups:
        versions = [sample.oldest_weight_version for sample in group
                    if sample.oldest_weight_version is not None]
        if versions and min(versions) > current_version:
            raise ValueError('partial rollout contains a future policy version')
        (stale if versions and current_version - min(versions) > max_age else retained).append(group)
    return retained, stale


class QualifiedMathRollout(InferenceRolloutFn):
    async def _call_train(self, input):
        self.constructor_input.args.archlab_policy_version = input.weight_version
        max_age = int(os.environ.get('ARCHLAB_PARTIAL_MAX_POLICY_AGE', '2'))
        if max_age < 0 or input.weight_version is None:
            raise ValueError('partial rollout requires a published policy version and nonnegative age')
        retained, stale = split_stale_groups(self.data_source.buffer, int(input.weight_version), max_age)
        self.data_source.buffer[:] = retained
        path = os.environ.get('ARCHLAB_FIRST_ROLLOUT_MANIFEST')
        if path and input.rollout_id == 0:
            output = self._bootstrap(Path(path), input)
        else:
            output = await super()._call_train(input)
        output.metrics = {
            **(output.metrics or {}),
            'sampling/stale_groups_dropped': len(stale),
            'sampling/stale_response_tokens_dropped': sum(sample.response_length for group in stale for sample in group),
            'sampling/partial_groups_retained': len(self.data_source.buffer),
            'sampling/partial_response_tokens_retained': sum(
                sample.response_length for group in self.data_source.buffer for sample in group),
            'sampling/partial_new_response_tokens': generated_tokens_at_version(
                [sample for group in self.data_source.buffer for sample in group], input.weight_version),
        }
        return output

    def _verified_manifest(self, manifest_path):
        args = self.constructor_input.args
        manifest = json.loads(manifest_path.read_text())
        # Miles resolves an omitted --load to the HF parent directory.
        parent_load = args.load is None or Path(args.load).resolve() == Path(args.hf_checkpoint).resolve()
        if (args.start_rollout_id != 0 or not parent_load or not args.no_load_optim
                or not args.no_load_rng or manifest['source_optimizer_updates'] != 0):
            raise ValueError('bootstrap is only valid for a fresh, unupdated parent')
        for path, expected in manifest['file_sha256'].items():
            if sha256_file(path) != expected:
                raise ValueError(f'bootstrap source identity changed: {path}')
        model = json.loads((Path(args.hf_checkpoint) / 'config.json').read_text())
        if model['archlab']['complete_sha256'] != manifest['parent_complete_sha256']:
            raise ValueError('bootstrap parent differs from training parent')
        if sha256_file(Path(args.hf_checkpoint) / 'config.json') != manifest['model_config_sha256']:
            raise ValueError('bootstrap model/rollout configuration differs')
        if sha256_file(args.prompt_data) != manifest['prompt_data_sha256']:
            raise ValueError('bootstrap prompt data differs')
        return manifest

    def _bootstrap(self, manifest_path, input):
        from miles.ray.rollout.debug_data import _load_rollout_data_file

        args = self.constructor_input.args
        manifest = self._verified_manifest(manifest_path)
        samples, _ = _load_rollout_data_file(Path(manifest['rollout_path']))
        groups = defaultdict(list)
        for sample in samples:
            groups[sample.group_index].append(sample)
            if sample.oldest_weight_version != int(input.weight_version):
                raise ValueError('bootstrap policy version differs from the initial published policy')
            if not sample.rollout_log_probs or sample.rollout_routed_experts is None:
                raise ValueError('bootstrap requires original log-probabilities and routing replay')
        if len(groups) != args.rollout_batch_size or any(len(group) != args.n_samples_per_prompt for group in groups.values()):
            raise ValueError('bootstrap requires a complete training batch of complete prompt groups')
        if min(groups) < 0 or max(groups) >= len(self.data_source.dataset):
            raise ValueError('bootstrap group indices must belong to the first prompt epoch')
        expected = self.data_source.get_samples(max(groups) + 1)
        for index, group in groups.items():
            if any(sample.prompt != expected[index][0].prompt or sample.label != expected[index][0].label
                   for sample in group):
                raise ValueError('bootstrap prompt ordering differs from the current data source')
        return RolloutFnTrainOutput(samples=list(groups.values()), metrics={
            'qualification/replayed_parent_batch': 1,
            'qualification/source_response_limit': manifest['source_response_limit'],
        })

    async def _call_eval(self, input):
        path = os.environ.get('ARCHLAB_FIRST_ROLLOUT_MANIFEST')
        if not path or input.rollout_id != 0:
            return await super()._call_eval(input)
        from miles.ray.rollout.debug_data import _load_rollout_data_file
        from miles.utils.types import Sample

        args = self.constructor_input.args
        manifest = self._verified_manifest(Path(path))
        evaluation = manifest['initial_evaluation']
        if len(args.eval_datasets) != 1:
            raise ValueError('bootstrap evaluation requires the identical single held-out dataset')
        dataset = args.eval_datasets[0]
        if (dataset.name != evaluation['name']
                or sha256_file(dataset.path) != evaluation['prompt_data_sha256']
                or dataset.max_response_len != evaluation['max_response_len']
                or dataset.n_samples_per_eval_prompt != evaluation['n_samples_per_prompt']
                or dataset.temperature != evaluation['temperature']
                or dataset.top_p != evaluation['top_p']
                or dataset.top_k != evaluation['top_k']):
            raise ValueError('bootstrap evaluation settings differ from the recorded parent evaluation')
        samples, _ = _load_rollout_data_file(Path(evaluation['rollout_path']))
        return RolloutFnEvalOutput(data={dataset.name: {
            'samples': samples, 'rewards': [sample.get_reward_value(args) for sample in samples],
            'truncated': [sample.status == Sample.Status.TRUNCATED for sample in samples],
        }}, metrics={'qualification/replayed_parent_evaluation': 1})
