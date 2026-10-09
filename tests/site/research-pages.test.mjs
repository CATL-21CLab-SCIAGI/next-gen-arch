import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, readdir, stat} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {CHART, chartPoint, curvePath, checkpointAt, firstObservedCrossing,
  targetComparison, reasoningMetric, validateEvidence} from '../../docs/site/zh/evidence.mjs';

const site = new URL('../../docs/site/', import.meta.url);
const zh = new URL('zh/', site);
const html = await readFile(new URL('index.html', zh), 'utf8');
const learning = JSON.parse(await readFile(new URL('assets/learning-curves.json', zh)));
const reasoning = JSON.parse(await readFile(new URL('assets/reasoning.json', zh)));
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, String(actual));

test('curve contains the audited paired points, preserving actual token positions', () => {
  assert.equal(validateEvidence(learning, reasoning), true);
  for (const [column, name] of [[1, 'normal'], [2, 'simplicial']]) {
    const full = learning.series.all_saved_points[name];
    assert.equal(full.length, 70);
    const byToken = new Map(full);
    for (const row of learning.series.points) assert.equal(row[column], byToken.get(row[0]));
  }
  assert.deepEqual(checkpointAt(learning, 65), {
    tokens: 1e10, normal: .8095661401748657, simplicial: .801841676235199,
  });
  assert.throws(() => checkpointAt(learning, 66), RangeError);
  assert.throws(() => checkpointAt(learning, -1), RangeError);
  assert.throws(() => checkpointAt(learning, 1.5), RangeError);
  assert.equal(chartPoint(1e10, .8)[0], CHART.right);
});

test('1.70x is explicitly a fixed 10B reference, not both first crossings', () => {
  const result = targetComparison(learning, 'endpoint');
  assert.equal(result.mode, 'fixed_endpoint_reference');
  assert.equal(result.normalTokens, 1e10);
  assert.equal(result.simplicialTokens, 5898240000);
  close(result.savingPercent, 41.0176);
  assert.equal(result.ratio.toFixed(2), '1.70');
  assert.equal(firstObservedCrossing(learning.series.all_saved_points.normal, result.target)[0], 9568256000);
  assert.match(html, /10B 为普通组终点参照/);
});

test('CE targets use earliest recorded crossing despite non-monotonic losses', () => {
  const a = targetComparison(learning, 'ce_081');
  assert.equal(a.mode, 'both_first_observed');
  assert.equal(a.normalTokens, 9043968000);
  assert.equal(a.simplicialTokens, 5767168000);
  const b = targetComparison(learning, 'ce_082');
  assert.equal(b.normalTokens, 5242880000);
  assert.equal(b.simplicialTokens, 3801088000);
  assert.deepEqual(firstObservedCrossing([[1, .9], [2, .7], [3, .85], [4, .6]], .8), [2, .7]);
  assert.equal(firstObservedCrossing([[1, .9], [2, .7]], .5), null);
  assert.throws(() => firstObservedCrossing([], NaN), TypeError);
  assert.throws(() => targetComparison(learning, 'unknown'), RangeError);
});

test('AIME metrics use strict correctness and all generated tokens', () => {
  const n = reasoning.results.normal, s = reasoning.results.simplicial;
  assert.deepEqual([n.correct_responses, s.correct_responses], [19, 25]);
  assert.deepEqual([n.total_generated_tokens, s.total_generated_tokens], [12327575, 9774678]);
  const accuracy = reasoningMetric(reasoning, 'accuracy');
  close(accuracy.normal, 19 / 120 * 100);
  close(accuracy.simplicial, 25 / 120 * 100);
  const efficiency = reasoningMetric(reasoning, 'efficiency');
  close(efficiency.normal, 1.5412601424043253);
  close(efficiency.simplicial, 2.557629008341758);
  close(efficiency.simplicial / efficiency.normal, 1.659440180132034);
  assert.equal(reasoning.protocol.problems, 30);
  assert.equal(reasoning.protocol.responses_per_problem, 4);
  assert.equal(reasoning.protocol.native_context_tokens, 131072);
  assert.throws(() => reasoningMetric(reasoning, 'unknown'), RangeError);
});

test('static fallback contains the same real curves and necessary evidence context', () => {
  for (const column of [1, 2]) assert.ok(html.includes(curvePath(learning.series.points, column)));
  assert.match(html, /固定验证/);
  assert.match(html, /2\.57%/);
  assert.match(html, /能力优势仍需复测/);
  assert.match(html, /科学发现闭环是下一步研究方向/);
});

test('anchors, accessible controls, and assets work under a project subpath', async () => {
  assert.match(html, /<html lang="zh-CN">/);
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
  assert.equal(new Set(ids).size, ids.length, 'IDs must be unique');
  for (const [, id] of html.matchAll(/href="#([^"]+)"/g)) assert.ok(ids.includes(id), id);
  for (const [, reference] of html.matchAll(/(?:href|src)="(\.[^"]+)"/g)) {
    assert.ok((await stat(new URL(reference, zh))).isFile(), reference);
  }
  assert.doesNotMatch(html, /(?:href|src)="\/[^/]/, 'Avoid root-absolute URLs on GitHub project Pages');
  for (const id of ['interaction-range', 'token-range']) {
    assert.match(html, new RegExp('for="' + id + '"'));
    assert.match(html, new RegExp('id="' + id + '"[^>]*type="range"'));
  }
  assert.match(html, /id="motion-toggle"[^>]*aria-label=/);
  const root = await readFile(new URL('index.html', site), 'utf8');
  assert.match(root, /url=\.\/zh\//);
});

test('published tree contains only reviewed static assets and compact aggregates', async () => {
  const allowed = new Set([
    '.nojekyll', 'index.html', 'zh/index.html', 'zh/styles.css', 'zh/favicon.svg',
    'zh/app.mjs', 'zh/evidence.mjs', 'zh/assets/learning-curves.json',
    'zh/assets/reasoning.json', 'zh/assets/particle-field.png', 'zh/assets/action-landscape.jpg',
  ]);
  const base = fileURLToPath(site);
  for (const name of await readdir(site, {recursive: true})) {
    const info = await stat(new URL(name, site));
    if (!info.isFile()) continue;
    assert.ok(allowed.delete(name), 'Unexpected published file: ' + name);
    assert.ok(info.size < 5 * 1024 * 1024, 'Oversized file: ' + name);
    if (/\.(html|css|mjs|json|svg)$/.test(name)) {
      const text = await readFile(base + name, 'utf8');
      assert.doesNotMatch(text, /(?:\/mnt\/|\/tmp\/|BEGIN [A-Z ]*PRIVATE KEY|ghp_[a-zA-Z0-9]{30})/);
      assert.doesNotMatch(text, /<script[^>]+src="https?:/);
    }
  }
  assert.equal(allowed.size, 0);
});
