/** Calculations use recorded checkpoints and observed response counts only. */
export const CHART = Object.freeze({left: 70, right: 972, top: 26, bottom: 391, xmin: 2e9, xmax: 1e10, ymin: .798, ymax: .865});

export function chartPoint(tokens, loss) {
  return [
    CHART.left + (tokens - CHART.xmin) / (CHART.xmax - CHART.xmin) * (CHART.right - CHART.left),
    CHART.bottom - (loss - CHART.ymin) / (CHART.ymax - CHART.ymin) * (CHART.bottom - CHART.top),
  ];
}

export function curvePath(points, column) {
  return points.map((row, index) => {
    const [x, y] = chartPoint(row[0], row[column]);
    return (index ? 'L' : 'M') + x.toFixed(3) + ',' + y.toFixed(3);
  }).join(' ');
}

export function checkpointAt(data, index) {
  if (!Number.isInteger(index) || index < 0 || index >= data.series.points.length) {
    throw new RangeError('Checkpoint index is outside the observed series');
  }
  const [tokens, normal, simplicial] = data.series.points[index];
  return {tokens, normal, simplicial};
}

export function firstObservedCrossing(points, target) {
  if (!Number.isFinite(target)) throw new TypeError('CE target must be finite');
  return points.find((point) => point[1] <= target) ?? null;
}

export function targetComparison(data, key) {
  const target = {endpoint: data.series.points.at(-1)[1], ce_081: .81, ce_082: .82}[key];
  if (target === undefined) throw new RangeError('Unknown target preset');
  const full = data.series.all_saved_points;
  const simplicial = firstObservedCrossing(full.simplicial, target);
  const normal = key === 'endpoint' ? full.normal.at(-1) : firstObservedCrossing(full.normal, target);
  if (!normal || !simplicial) throw new RangeError('Both observed crossings are required');
  return {
    target,
    mode: key === 'endpoint' ? 'fixed_endpoint_reference' : 'both_first_observed',
    normalTokens: normal[0],
    simplicialTokens: simplicial[0],
    ratio: normal[0] / simplicial[0],
    savingPercent: (1 - simplicial[0] / normal[0]) * 100,
  };
}

export function reasoningMetric(data, mode) {
  if (!['accuracy', 'efficiency'].includes(mode)) throw new RangeError('Unknown reasoning metric');
  const score = (result) => mode === 'accuracy'
    ? result.correct_responses / result.responses * 100
    : result.correct_responses / result.total_generated_tokens * 1e6;
  return {normal: score(data.results.normal), simplicial: score(data.results.simplicial), ceiling: mode === 'accuracy' ? 30 : 3};
}

export function validateEvidence(data, reasoning) {
  const points = data.series.points;
  if (points.length !== 66 || points.some((row, i) => row.length !== 3
    || row.some((v) => !Number.isFinite(v)) || row[2] >= row[1]
    || (i > 0 && row[0] <= points[i - 1][0]))) throw new Error('Invalid paired curve evidence');
  if (points.at(-1)[0] !== 1e10 || data.comparison.training_seed_pairs !== 1) {
    throw new Error('Unexpected experiment contract');
  }
  for (const variant of ['normal', 'simplicial']) {
    const result = reasoning.results[variant];
    if (result.responses !== 120 || result.total_generated_tokens <= 0
      || result.correct_responses < 0 || result.correct_responses > result.responses) {
      throw new Error('Invalid observed evaluation counts');
    }
  }
  if (reasoning.protocol.native_context_tokens !== 131072) throw new Error('Unexpected evaluation context');
  return true;
}
