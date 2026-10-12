# Prepared agentic SFT data

`archlab.data.agentic_sft` prepares text-only, bash-only trajectories for the
deployed MiMo `BashOnlyAgent` and native tokenizer. It does not execute recorded
commands or task verifiers. Upstream verl owns training and next-token shifting.

`convert_mini_swe_messages` accepts complete mini-swe trajectories with one
fenced bash command per assistant turn, a PR description, complete return-code
and output observations, and the exact terminal submission sentinel. It removes
the source agent's fenced-action instructions, projects commands into structured
`bash` calls, rebases an explicitly supplied working-directory prefix, and uses
the supplied runtime observation formatter. The submission sentinel plus
staging/diff command becomes the source's final assistant thought, with no tool
call, matching the deployed agent's idle termination. Truncated observations,
ambiguous actions, other tools, and incomplete links fail closed.

`encode_assistant_targets` renders through the actual `MessageCodec` message and
tool projection and the unmodified native chat template. It verifies that every
assistant prefix equals the gateway's generation prompt. Assistant content and
native turn EOS receive loss; the system, user, observation, role headers and
separators receive none. Overlength trajectories are rejected. The Limite native
template renders tool observations as user `<tool_response>` blocks; preparation
does not introduce a separate text protocol.

The Parquet schema has four variable-length `list<int64>` columns:

| Column | Contract |
| --- | --- |
| `input_ids` | Native rendered conversation; no padding or truncation |
| `attention_mask` | All ones |
| `loss_mask` | Binary, **unshifted target-token** alignment; first token zero |
| `position_ids` | Contiguous from zero |

`PreparedAgenticSFTDataset(parquet_files, tokenizer, config, processor=None,
max_samples=-1)` reads this format and returns four `torch.int64` tensors. It
validates lengths, padding, positions, nonempty assistant loss, and supervised
native EOS. Set upstream `data.custom_cls.path=pkg://archlab.data.agentic_sft`
and `data.custom_cls.name=PreparedAgenticSFTDataset`. Use upstream
`data.pad_mode=no_padding` with context limits admitted against the manifest;
the dataset never shifts the loss mask itself.

## Public bootstrap receipt, 2026-10-13

The pinned public source is
[JetBrains-Research/agent-trajectories-swesmith-random-subset](https://huggingface.co/datasets/JetBrains-Research/agent-trajectories-swesmith-random-subset/tree/796cd9f45734efebaf92ad0b8c27b975e1225e6e),
revision `796cd9f45734efebaf92ad0b8c27b975e1225e6e`, MIT. Its card documents
mini-swe-agent traces, a per-step router between GPT-5.2 and GPT-5-mini, and a
`resolved` column from SWE-bench evaluation. Selection requires both
`resolved=true` and `exit_status=Submitted`; preparation does not repeat that
evaluation. No teammate trajectories are used.

The small bootstrap contains 149 trajectories: 135 train and 14 validation,
67 repositories, 1,034 bash calls, 184,005 assistant target tokens, and 1,183
supervised native EOS tokens. Total input is 1,073,017 tokens; lengths range
from 2,297 to 16,070 under a **16,384-token** cap. Every selected assistant turn
passes native prefix and Hermes parser roundtrip checks. Validation task IDs
are disjoint from training; repositories are shared, so this split does not
measure generalization to new repositories.

All 11 distinct MiMo validation tasks from the original eight-task pilot and
the current three-task pilot are excluded by ID and exact problem hash.
Those MiMo records do not expose upstream commit/repository metadata and use
opaque task IDs. The receipt therefore records repository identity evidence
from each heldout description/test module and restricts the public data to an
explicit, reviewed repository allowlist. The JavaScript GitHub-action task's
owner is unavailable; the Python-library allowlist excludes its project class.
The receipt does not claim an exact mapping between opaque MiMo and synthetic
SWE-Smith IDs.

The native snapshot is `/mnt/oss/models/limite-1b-base-cc612bafcd4a`:
publisher generation EOS is `151645` (`<|im_end|>`), while tokenizer EOS/BOS is
`151643` (`<|endoftext|>`). Neither tokenizer assets nor template are modified.

Reproduction script and logs are in the ignored
`results/agentic-conversion-20261013/` directory. `bootstrap/manifest.json`
records source and native asset hashes, runtime code hashes, protocol,
exclusions, source rejection counts, sampling seed/order, coverage, token
counts and prepared-file hashes. `index.jsonl` records per-task source row,
repository/commit prefix, source-message hash, model selections and token
counts. `prepared_sample.json` shows a projected native conversation.

The maintained module is exercised by `tests/test_agentic_sft.py`. Set
`ARCHLAB_NATIVE_TOKENIZER` to the qualified local snapshot to run the native
codec integration test together with `tests/test_native_agentic_contract.py`;
these tests require the installed upstream agent runtime but no GPU.
