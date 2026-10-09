# Chinese research page

The public research narrative is at
[从炼金术到材料学](https://catl-21clab-sciagi.github.io/next-gen-arch/zh/).
Source lives in `docs/site/`; GitHub Pages publishes only that directory.
It uses plain HTML, CSS, SVG and JavaScript modules, without a build framework
or third-party runtime requests.

## Editorial and evidence contract

The narrative and two illustrations are adapted from Changqing Fu's supplied
Chinese presentation, *从炼金术到材料学*. Its operational and retrospective
sections (pages 44–50) are not reproduced. The page introduces the research
program and selected observations; the Wiki retains the full methodology.
Scientific-discovery workflows are identified as future research, with links
to the organization's public research, evaluation and training foundations.

The learning chart uses 66 shared saved checkpoints plus the 70 saved points
per variant for threshold calculations. These are fixed-validation CE on
32 windows of 2,048 targets, not the training objective or a full-corpus
validation estimate. The underlying comparison and provenance are in
[Limite SFT comparison](wiki/Limite-SFT-Comparison.md).

The default 1.70× token comparison uses the ordinary model's **fixed 10B
endpoint** as its reference and the simplicial model's first observed saved
checkpoint meeting that CE (5.89824B). The normal model itself first reaches
that CE at 9.568256B. The two selectable absolute CE targets compare both
variants' first observed crossings. No interpolation or monotonicity
assumption is used.

AIME 2026 uses strict correctness (19/120 and 25/120), the native 131,072-token
total context, and all generated tokens in the efficiency denominator.
The displayed 1.66× is one observed answer-per-token ratio, not an established
capability or wall-clock speedup. The single training pair, extra 2.57% total
parameters, and need for replication remain visible.

Only sanitized aggregates are published; no prompts, raw completions, queue
state, infrastructure receipts, or model payloads belong in this directory.

## Updating and publishing

Run from the repository root with Node.js 24:

    node --check docs/site/zh/app.mjs
    node --check docs/site/zh/evidence.mjs
    node --test tests/site/*.test.mjs

When changing the JSON, update the static HTML fallback from the same data.
The tests verify both paths, the audited numeric anchors, and project-relative
asset links. The page remains readable without JavaScript. Its controls use
native buttons and sliders; motion can be paused and respects the system's
reduced-motion setting.

The `Chinese research page` workflow validates relevant pull requests and
deploys on `main`. Publication uses the repository's native GitHub Pages
environment, not a separate repository. The root URL redirects to `zh/`.
