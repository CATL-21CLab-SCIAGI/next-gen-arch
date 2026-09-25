# Documentation

**Current DeepSeek RL path:** [Miles baseline](MILES_BASELINE.md). This is the maintained entry for launch commands, runtime pins, custom hooks, and qualification status.

The reader guides are checked in under **[`wiki/`](wiki/Home.md)**. These page sources can also be published to the [GitHub Wiki](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/wiki).

**Publication status:** Wiki is enabled, but its Git repository returned “repository not found” on 2026-09-23. Create the initial Home page on GitHub before publishing this prepared page set. Until then, start with the [Home source](wiki/Home.md).

## Reader routes

| Goal | Page source |
| --- | --- |
| Understand the project | [Research Program](wiki/Research-Program.md) |
| Find code ownership | [Code Map](wiki/Code-Map.md) |
| Design an experiment | [Experiment Design](wiki/Experiment-Design.md) |
| Select an execution path | [Runtime and Backends](wiki/Runtime-and-Backends.md) |
| Launch or resume | [Runbook](wiki/Runbook.md) |
| Prepare data and artifacts | [Data and Provenance](wiki/Data-and-Provenance.md) |
| Compare capabilities | [Evaluation](wiki/Evaluation.md) |
| Read conclusions | [Results](wiki/Results.md) |
| Review model lineages | [Qwen](wiki/Qwen-Experiments.md), [DeepSeek](wiki/DeepSeek-Experiments.md), [Math RL](wiki/Math-RL.md) |
| Monitor runs | [Tracking and Storage](wiki/Tracking-and-Storage.md) |
| Contribute | [Contributing](wiki/Contributing.md), [Documentation Maintenance](wiki/Documentation-Maintenance.md) |

## Versioned records

| Category | Entry |
| --- | --- |
| Historical sweeps and backend comparisons | [Results index](RESULTS.md) |
| Qwen/DeepSeek evidence audit | [2026-09-22 conclusions](TRAINING_CONCLUSIONS_20260922.md) |
| Upstream optimization ideas and dispositions | [Optimization audit](OPTIMIZATION_AUDIT.md) |
| Scientific contracts | [Experiment contracts](EXPERIMENT_CONTRACTS.md) |
| Compact source evidence | [Recorded results](recorded-results/README.md) |

Dated reports describe their recorded experiment. They are not live runbooks. Local `results/` paths identify team-storage artifacts; they are not GitHub downloads.

## Publish the Wiki

After creating its first page on GitHub, use a separate Wiki checkout:

```bash
git clone https://github.com/CATL-21CLab-SCIAGI/next-gen-arch.wiki.git /path/to/next-gen-arch.wiki
cp docs/wiki/*.md /path/to/next-gen-arch.wiki/
git -C /path/to/next-gen-arch.wiki diff --check
git -C /path/to/next-gen-arch.wiki diff
```

Review existing Wiki edits, then commit and push the approved page changes in that checkout. Preserve unrelated Wiki pages. `_Sidebar.md` and `_Footer.md` supply navigation.
