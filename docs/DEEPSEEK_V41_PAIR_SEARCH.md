# Modular pair-search probes

**Status:** exploratory design; no comparative result is established here.

## Task

Choose one entry from A and one from B whose two coordinates sum to query T modulo 97.

| Property | Design |
| --- | --- |
| List sizes | A=2; B=8/16/32/64 |
| NO cases | Each A entry matches each coordinate in different B entries |
| YES twin | Swap two B second coordinates to create exactly one solution |
| Preserved marginals | Coordinate multisets/frequencies, A and T |
| Scoring | Exact stripped YES/NO; report format failures separately |

A complement lookup solves the task efficiently. This probe does not establish a complexity separation for deep language models.

## Generation

```bash
PYTHONPATH=src python -m archlab.evaluation.pair_search \
  --output /path/to/new-pair-search.yaml
```

Templates live in `src/archlab/prompts/deepseek_v41_pair_search_templates_v1.yaml`; the versioned sample is `deepseek_v41_pair_search_v1.yaml`.

## Evaluation rules

1. Keep twins in the same split; send only prompt text to the model.
2. Calibrate difficulty on separate examples without selecting for a favorable variant gap.
3. Include specified-pair arithmetic controls and near-ceiling controls.
4. Audit native token positions against the adapter's 32/512 windows; character swaps do not guarantee equal token lengths.
5. Compare exact matched checkpoint identities and decoding settings.
6. Report balanced accuracy, size breakdown and both-twins-correct rate; bootstrap by twin pair.

Both models at chance is uninformative. Hidden-state context and multiple layers prevent a literal token-level causal interpretation.

[Motivating modular matching work](https://arxiv.org/html/2306.02896v2) · [Evaluation guide](wiki/Evaluation.md)
