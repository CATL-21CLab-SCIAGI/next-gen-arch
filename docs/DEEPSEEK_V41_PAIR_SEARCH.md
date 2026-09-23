# Exploratory modular pair-search probes

These probes ask whether one entry from A and one from B sum to a query T,
coordinate by coordinate modulo 97. They target query-conditioned pair binding;
they do not establish a complexity separation for our deep pretrained models.
A complement lookup algorithm also solves this task efficiently. No exhaustive
quadratic computation is inherently required.

The default ladder has two entries in recent list A and 8, 16, 32, or 64 in
earlier list B. Every B entry satisfies at least one coordinate for some A
entry. In every NO case, each A entry has matches for both coordinates, but
in different B entries. A YES twin swaps two B second coordinates, creating
exactly one solution. Both coordinate multisets, their frequencies, A, and T
are unchanged. These are difficult-binding candidates; model difficulty has
not yet been measured. Swaps do not guarantee identical tokenizer lengths.

Prompt text lives in `src/archlab/prompts/deepseek_v41_pair_search_templates_v1.yaml`.
The eight-case sample lives in `src/archlab/prompts/deepseek_v41_pair_search_v1.yaml`.
Regenerate it from the repository root:

```sh
PYTHONPATH=src python -m archlab.evaluation.pair_search \
  --output src/archlab/prompts/deepseek_v41_pair_search_v1.yaml
```

Use another seed and `--pairs-per-size 100` for 800 new cases. Keep both twins
in the same calibration/evaluation split. Send only each row's `text`, never
the reference answer, witness indices, ID, tags, or other generator metadata.
Run each prompt as an independent conversation, with the same native
non-thinking encoder and decoding policy for both checkpoint variants.

Before attributing results to the windowed adapter, audit token positions
after native chat formatting: the short axis reaches 32 tokens and the long
axis 512, both including the query position. A is placed near the answer to
help satisfy these constraints, but eligibility is not verified here. Larger
lists, extra recent entries, native conversation suffixes, and larger moduli
can exceed the windows. Record any such cases as separate context stress
tests. Contextual hidden states and multilayer computation still prevent a
literal token-level causal interpretation even inside both windows.

Calibrate difficulty on separate cases against the baseline without selecting
settings for a favorable simplicial gap. Freeze a ladder spanning near-ceiling
through intermediate performance before comparative evaluation. Both models
at chance is uninformative. Preserve near-ceiling levels as controls. Add
specified-pair arithmetic controls to distinguish search errors from modular
arithmetic errors before making a mechanism claim.

Compare actual 4537/4537 checkpoint fingerprints. Score stripped outputs by
exact YES/NO match, counting format failures as incorrect while also reporting
them separately. Report balanced accuracy, accuracy by list size, and the
fraction of twin pairs for which both answers are correct. Bootstrap by twin
pair, since twins are correlated. Increasing the modulus changes arithmetic
difficulty as well as search difficulty and should be a separate factor.

This suite has not been sent to either model. It is inspired by modular triple
matching in https://arxiv.org/html/2306.02896v2, but neither the language encoding
nor the two-list/two-coordinate adaptation inherits that paper's theorem.
