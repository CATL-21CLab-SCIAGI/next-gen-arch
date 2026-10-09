import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, readdir, stat} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {CHART, chartPoint, curvePath, checkpointAt, checkpointEfficiency, firstObservedCrossing,
  targetComparison, validateEvidence, sceneProgress, scrollCheckpointIndex} from '../../docs/site/zh/evidence.mjs';

const site = new URL('../../docs/site/', import.meta.url);
const zh = new URL('zh/', site);
const html = await readFile(new URL('index.html', zh), 'utf8');
const learning = JSON.parse(await readFile(new URL('assets/learning-curves.json', zh)));
const warmup = JSON.parse(await readFile(new URL('assets/warmup.json', zh)));
const internet = JSON.parse(await readFile(new URL('assets/internet-data.json', zh)));
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, String(actual));

test('curve contains the audited paired points, preserving actual token positions', () => {
  assert.equal(validateEvidence(learning, warmup), true);
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

test('scroll efficiency follows the current normal budget and an observed simplicial crossing', () => {
  const first = checkpointEfficiency(learning, 0);
  assert.equal(first.ratio, 1);
  assert.equal(first.simplicialTokens, 2000158720);
  const last = checkpointEfficiency(learning, 65);
  assert.equal(last.target, .8095661401748657);
  assert.equal(last.normalTokens, 1e10);
  assert.equal(last.simplicialTokens, 5898240000);
  assert.equal(last.simplicialLoss, .8090752363204956);
  close(last.ratio, 1.6954210069444444);
  for (let i = 0; i < learning.series.points.length; i++) {
    const result = checkpointEfficiency(learning, i);
    assert.equal(result.normalTokens, checkpointAt(learning, i).tokens);
    assert.ok(result.simplicialLoss <= result.target);
    assert.ok(result.simplicialTokens <= result.normalTokens);
    for (const [tokens, loss] of learning.series.all_saved_points.simplicial) {
      if (tokens >= result.simplicialTokens) break;
      assert.ok(loss > result.target, 'No earlier observed match may be skipped');
    }
  }
  const fixture = {series: {
    points: [[10, .8, .7]],
    all_saved_points: {simplicial: [[2, .9], [4, .78], [5, .85], [9, .72], [10, .7]]},
  }};
  assert.deepEqual(checkpointEfficiency(fixture, 0), {
    target: .8, normalTokens: 10, simplicialTokens: 4, simplicialLoss: .78, ratio: 2.5,
  });
  fixture.series.all_saved_points.simplicial = [[2, .9]];
  assert.throws(() => checkpointEfficiency(fixture, 0), RangeError);
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

test('published AIME evidence is adapter-only warmup against base, without RL', () => {
  assert.equal(warmup.stage, 'adapter_only_warmup');
  assert.equal(warmup.training.backbone_frozen, true);
  assert.equal(warmup.training.actual_supervised_targets, 2000158720);
  assert.equal(warmup.training.rl_updates, 0);
  for (const [variant, correct] of [['base', 5], ['normal', 25], ['simplicial', 27]]) {
    const result = warmup.results[variant];
    assert.equal(result.correct_responses, correct);
    assert.equal(result.responses, 120);
    close(result.mean_pass_at_1, correct / 120);
    assert.equal(result.rl_updates, 0);
  }
  assert.equal(warmup.protocol.problems, 30);
  assert.equal(warmup.protocol.responses_per_problem, 4);
  assert.equal(warmup.protocol.native_context_tokens, 131072);
  assert.equal(warmup.protocol.grader, 'strict_answer_grader');
  assert.match(html, /4\.2<span>%/);
  assert.match(html, /20\.8<span>%/);
  assert.match(html, /22\.5<span>%/);
  assert.doesNotMatch(html, /1\.66|15\.8|RL400|reasoning\.json/);
});

test('scrolling spans all saved checkpoints and reverses without extrapolation', () => {
  assert.equal(sceneProgress(200, 2000, 700, 66), 0);
  assert.equal(sceneProgress(66, 2000, 700, 66), 0);
  assert.equal(sceneProgress(-584, 2000, 700, 66), .5);
  assert.equal(sceneProgress(-1234, 2000, 700, 66), 1);
  assert.equal(sceneProgress(-3000, 2000, 700, 66), 1);
  assert.equal(sceneProgress(0, 600, 700), 1);
  assert.equal(sceneProgress(NaN, 600, 700), 1);
  const indices = [0, .25, .5, 1, .5, 0].map((p) => scrollCheckpointIndex(learning, p));
  assert.deepEqual(indices, [0, 16, 33, 65, 33, 0]);
  assert.equal(scrollCheckpointIndex(learning, NaN), 65);
  assert.equal(scrollCheckpointIndex(learning, -1), 0);
  assert.equal(scrollCheckpointIndex(learning, 2), 65);
});

test('internet data wall preserves PDF data and distinguishes the illustrative fit', () => {
  assert.equal(internet.points.length, 7);
  assert.deepEqual(internet.points.map((p) => p.trillion_tokens), [.3, 1.4, 2, 15, 18, 14.8, 36]);
  const xs = internet.points.map((p) => p.year - 2020), ys = internet.points.map((p) => Math.log10(p.trillion_tokens));
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const mx = mean(xs), my = mean(ys);
  const slope = xs.reduce((sum, x, i) => sum + (x - mx) * (ys[i] - my), 0) / xs.reduce((sum, x) => sum + (x - mx) ** 2, 0);
  close(slope, internet.illustrative_fit.log10_tokens_slope_per_year);
  close(10 ** slope, internet.illustrative_fit.annual_growth);
  assert.equal(internet.stock.median_trillion_tokens, 300);
  assert.deepEqual(internet.stock.confidence_interval_90_percent, [100, 1000]);
  assert.match(html, /但互联网只有一个/);
  assert.match(html, /2028 年并非确定的耗尽日期/);
  for (const p of internet.points) assert.ok(html.includes(p.model));
});

test('static fallback contains the same real curves and necessary evidence context', () => {
  for (const column of [1, 2]) assert.ok(html.includes(curvePath(learning.series.points, column)));
  assert.match(html, /固定验证/);
  assert.match(html, /2\.57%/);
  assert.match(html, /差异仍需复测/);
  assert.match(html, /科学发现闭环是下一步研究方向/);
  const learningScene = html.slice(html.indexOf('id="learning"'), html.indexOf('class="learning-detail'));
  assert.match(learningScene, /id="checkpoint-ratio">1\.70×/);
  assert.match(learningScene, /id="checkpoint-budgets">10\.00B ÷ 5\.90B/);
  assert.match(learningScene, /首次达到或低于该 CE 的保存点 token；含预热，不作插值/);
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
  assert.doesNotMatch(html, /type="range"|comparison-handle|token-range|interaction-range/);
  for (const name of ['attention', 'learning']) assert.ok(html.includes('data-scroll-scene="' + name + '"'));
  assert.match(html, /id="motion-toggle"[^>]*aria-label=/);
  const root = await readFile(new URL('index.html', site), 'utf8');
  assert.match(root, /url=\.\/zh\//);
});

test('published tree contains only reviewed static assets and compact aggregates', async () => {
  const allowed = new Set([
    '.nojekyll', 'index.html', 'zh/index.html', 'zh/styles.css', 'zh/favicon.svg',
    'zh/app.mjs', 'zh/evidence.mjs', 'zh/assets/learning-curves.json',
    'zh/assets/warmup.json', 'zh/assets/internet-data.json', 'zh/assets/attention-illustrations.svg', 'zh/assets/particle-field.png', 'zh/assets/action-landscape.jpg',
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

test('attention scene uses original PDF artwork for both scroll states', async () => {
  assert.match(html, /attention-illustrations\.svg#pairwise/);
  assert.match(html, /attention-illustrations\.svg#simplicial/);
  const svg = await readFile(new URL('assets/attention-illustrations.svg', zh), 'utf8');
  assert.match(svg, /<view id="pairwise" viewBox="130 140 280 280"/);
  assert.match(svg, /<view id="simplicial" viewBox="550 140 280 280"/);
  assert.doesNotMatch(svg, /<script|https?:\/\/(?!www\.w3\.org|www\.inkscape\.org)/);
});
