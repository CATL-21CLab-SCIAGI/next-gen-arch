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

Run `tests/test_native_agentic_contract.py` in the package runtime with
`ARCHLAB_NATIVE_TOKENIZER` pointing to the local Limite Base snapshot. It uses
upstream BashOnlyAgent tool declarations, Hermes parsing and GatewaySession;
checks exact two-turn rendering, full native-context allowance, and masks tool
observations out of policy loss. Its backend replays recorded tokens; this test
neither executes task code nor establishes GPU inference or sandbox readiness.
