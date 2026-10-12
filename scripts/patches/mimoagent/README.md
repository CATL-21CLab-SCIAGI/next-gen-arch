# Local MiMo-Agent patches

Apply the patches in filename order inside the pinned
`third_party/verl/third_party/mimoagent-osr` checkout.

```sh
git -C third_party/verl/third_party/mimoagent-osr apply "$PWD/scripts/patches/mimoagent/0001-report-context-exhaustion.patch"
git -C third_party/verl/third_party/mimoagent-osr apply "$PWD/scripts/patches/mimoagent/0002-anchor-cleanup-dependency-paths.patch"
git -C third_party/verl/third_party/mimoagent-osr apply "$PWD/scripts/patches/mimoagent/0003-preserve-generated-python-versions.patch"
```

The OpenAI client preserves the server's finish reason outside chat payloads.
The existing agent termination mechanism reports a nonempty length-limited
response as `LimitsExceeded`, retains generated history and avoids dispatching
partial tool calls. The existing verifier can still grade earlier valid edits;
context exhaustion does not manufacture a zero reward or a completed episode.

`tests/test_mimo_truncation_contract.py` exercises the actual client and BashOnlyAgent.

The second patch anchors dependency exclusions to the repository root. With
upstream cleanup enabled, `lib/` stays available while `build/lib/` is removed;
an unqualified `lib` pattern accidentally preserves that build residue.
`tests/test_mimo_cleanup_contract.py` checks the real cleanup in a disposable Git
tree, including preservation of root dependencies and `Manifest.toml`.

`0003-preserve-generated-python-versions.patch` preserves explicit generated version files from the captured base commit's Hatch-VCS/setuptools-scm configuration. Only regular, literal repository-relative files qualify; agent edits, globs, directories and traversal never expand cleanup exclusions. Original bytes stay in parent-side environment state and are restored before grading; changed leaf or parent symlink targets are rejected. This keeps offline editable installs such as tox runnable without admitting policy edits to untracked generated metadata.
