# Local Uni-Agent patches

Target: XiaomiMiMo/verl's nested `third_party/uni_agent` revision
`c63e0b01c375ebede95e01fe92bc367df24e5bf3`.

Apply these patches in order inside `third_party/verl/third_party/uni_agent`:

```sh
for patch in scripts/patches/uni-agent/*.patch; do
  git -C third_party/verl/third_party/uni_agent apply "$PWD/$patch"
done
```

- `0001-decode-json-tool-history.patch`: decode ordinary JSON-string function arguments before native chat rendering. This preserves the same argument object after tool execution and full-history reconstruction.
- `0002-use-generation-stop-contract.patch`: forward the existing HF model generation configuration's EOS into the gateway as the authoritative stop contract, using tokenizer EOS only when generation EOS is absent. Explicit `ignore_eos` prevents vLLM from adding a conflicting tokenizer EOS back. Strip accepted terminators from tool-call content. Publisher tokenizer/configuration files are never rewritten.
- `0003-record-rollout-identity.patch`: add the prompt UID to session manifests so each consumed `{uid}_{session}_{trajectory}` key maps to its exact token/mask/verifier artifacts, without counting unrelated asynchronous prefetch.

- `0004-validate-trajectory-metadata.patch`: optional typed boundary validator checks each original trajectory before TransferQueue transport and attaches task/group/session, policy-version, termination and EOS metadata without replacing upstream token transport.
- `0005-preserve-length-finish-reason.patch`: retain backend `length` / `max_tokens` termination when earlier complete tool calls parse. Keep their message and tokens for history, while the gateway's OpenAI response lets MiMo report `LimitsExceeded` before dispatching any calls from that truncated turn. EOS-completed tool turns still report `tool_calls`.
- `0006-native-muse-codec.patch`: detect templates requiring nested function-call history, resolve nameless tool observations from their linked calls during incremental encoding, and use the installed vLLM Muse reasoning/tool parsers. Generated token histories remain intact, including interleaved reasoning; malformed tool channels receive normal harness feedback and capped turns retain `length`. This requires the native Muse package runtime, not a parser backport.

Run `tests/test_native_agentic_contract.py` in the package runtime with
`ARCHLAB_NATIVE_TOKENIZER` pointing to the local Limite Base snapshot. It uses
upstream BashOnlyAgent tool declarations, Hermes parsing and GatewaySession;
checks exact two-turn rendering, full native-context allowance, and masks tool
observations out of policy loss. Its backend replays recorded tokens; this test
neither executes task code nor establishes GPU inference or sandbox readiness.
The same test file also checks the native tokenizer, Hermes parser, gateway,
OpenAI serialization and MiMo length guard together: capped turns retain their
parsed calls and tokens but execute no actions; EOS-completed turns execute one
mocked action. Run it with `pytest -q tests/test_native_agentic_contract.py`.

For Muse, set `ARCHLAB_MUSE_TOKENIZER` to the checksum-verified publisher
snapshot and run `tests/test_muse_agentic_codec.py` in the Muse runtime. It
checks nested history, linked observations, native EOS, quoted tool examples,
malformed/capped calls, and exact retained generated tokens with observation
loss masks. A semantic dictionary cannot represent every possible interleaving
of Muse channels; only a representable reasoning/call pair is additionally
compared with a complete native template rendering.

`0007-pin-framework-worker-to-driver-node.patch` places the rollout controller
(`AgentFrameworkWorker`) on the node that creates it. On an external Ray head,
evictable CPU agent pods join the cluster; losing the controller there ended
the run (2026-10-20/21), while losing a pod now only loses its runner tasks.
`archlab.training.runtime` pins the TaskRunner the same way
(`tests/test_training_runtime.py`).
