# Chinese research page

The public research narrative is at
[从炼金术到材料学](https://catl-21clab-sciagi.github.io/next-gen-arch/zh/).
Source lives in docs/site/; GitHub Pages publishes only that directory.
The page uses HTML, CSS, SVG and JavaScript modules, without a build framework
or third-party runtime requests.

## Editorial and evidence contract

The narrative, original headings and illustrations are adapted from
Changqing Fu's supplied presentation, *从炼金术到材料学*. The internet-data
slide is reconstructed as a responsive vector chart using its seven reported
model data points and illustrative log-linear fit. The roughly 2028 crossing
is a conditional extrapolation, not a definite data-exhaustion date. The
quality/repetition-adjusted stock estimate comes from Epoch AI (2024).
Reported model training tokens include different sources and are not a
measurement of unique human text consumption.

The public AIME comparison is limited to **base versus adapter-only 2B
warmup**: 5/120, 25/120 and 27/120 strictly correct responses. The backbone
remains frozen and no RL updates have occurred. All three use the same
problems and sample seeds, temperature 0.6, top-p 0.95 and native 131,072-token
total context. This is motivation for further work, not proof that either
adapter is architecturally superior. Preliminary 10B SFT and RL AIME outcomes
and their efficiency ratios are omitted from the presentation and current
site assets. Full methodology remains in the Wiki.

The learning chart still covers the 2B–10B supervised continuation: 66 paired
saved checkpoints and 70 points per variant for threshold calculations.
These are fixed-validation CE on 32 windows of 2,048 targets, not full-corpus
validation or independent reasoning scores. The default 1.70× comparison uses
the ordinary model's fixed 10B endpoint and the first observed simplicial
checkpoint meeting that CE (5.89824B). Normal first reaches that CE at
9.568256B. The other two CE presets compare both first observed crossings.
No interpolation or monotonicity assumption is used.

The parameter difference (2.57%) and limited training replication remain
visible. Scientific-discovery workflows are identified as future research,
with links to the organization's existing public foundations. The operational
and retrospective presentation sections (pages 44–50) are not reproduced.

Only sanitized aggregates are published. No prompts, raw completions,
queue state, infrastructure receipts, or model payloads belong here.

## Interaction and publication

The attention comparison uses the original vector artwork from PDF page 20,\nwith named SVG views for the two diagrams so the source artwork is stored once.\nPage scrolling drives the attention comparison and saved-checkpoint cursor;
there are no manual drag handles. Keyboard and touch page scrolling work
through the native document. Discrete component and CE-preset buttons remain.
Motion can be paused and respects reduced-motion preferences. Static figures
and the final learning snapshot remain readable without JavaScript.

Run from the repository root with Node.js 24:

    node --check docs/site/zh/app.mjs
    node --check docs/site/zh/evidence.mjs
    node --test tests/site/*.test.mjs

When changing JSON evidence, update the static SVG/HTML from the same data.
When changing the page's DOM or JavaScript/CSS contract, bump the shared asset
version query in the HTML and module import so cached assets cannot mix revisions.
Tests verify numeric anchors, static fallbacks, scroll mapping, and project
asset paths. The Chinese research page workflow validates pull requests and
deploys only docs/site/ from main to the repository's native GitHub Pages.
The root URL redirects to zh/.
