# Higher-Order Attention Under Matched Training

This working manuscript synthesizes recorded ordinary-versus-2-simplicial
experiments. It centers on the Limite comparison, includes the later aligned
scratch prefixes, and retains earlier negative Qwen/DeepSeek controls. It is
not a report of new model evaluations or a claim of architectural superiority.

## Build

The maintained manuscript is `manuscript.html.j2`. The builder reads compact
evidence, checks the reported arithmetic, generates five Matplotlib figures,
and embeds their SVGs into one standalone HTML document. It uses NumPy,
Matplotlib, and Jinja2; the exact build versions are written to the manifest.
Display equations are written in TeX notation and typeset by Matplotlib's
Mathtext engine into embedded vector paths. This avoids missing math glyphs
in browser PDF exports and needs no external fonts or system TeX installation.

From the repository root:

```bash
PYTHONPATH=src python -m archlab.reporting.attention_paper \
  --output results/attention-paper-20261013
```

Open `results/attention-paper-20261013/index.html`. All content and figures
render without a network connection. Links to references require the internet;
links to CSV/JSON and figure exports require the companion output directory.
The optional `paper.pdf` is a browser print export of the same HTML.
To recompile it, install Playwright and its Chromium runtime, then add `--pdf`
to the build command. `--browser-executable /path/to/chromium` selects an
existing browser. The export verifies that all five equation rows are already
typeset, waits for fonts, and records HTML/PDF hashes in
`data/pdf-verification.json`.

Generated figures are available as SVG, PDF, and PNG, alongside their CSV data.
`data/manifest.json` records source identities and checksums;
`data/verification.json` records the numerical checks. The output belongs in
ignored `results/`, not in the deployed research website.

## Evidence policy

The versioned `data/` files contain compact measurements and source checksums,
not raw responses, model payloads, private endpoints, or live job state.
Existing Limite aggregates remain authoritative in `docs/site/zh/assets/` and
`docs/recorded-results/`; the builder reads them in place instead of maintaining
another tracked copy.

Keep the manuscript and plots aligned when changing evidence. Preserve the
distinction between one training pair and repeated evaluation points, between
diagnostic CE and generated-answer accuracy, and between token and measured
compute efficiency. Unfinished runs remain unfinished in captions and tables.
Independent GPU qualification, training replication, and a repaired data
comparison are outside this document build.
