import {CHART, chartPoint, curvePath, checkpointAt, checkpointEfficiency, targetComparison, validateEvidence, sceneProgress, scrollCheckpointIndex} from './evidence.mjs?v=20261010-3';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const wiki = 'https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/wiki/';
const motionMedia = window.matchMedia('(prefers-reduced-motion: reduce)');
let userPaused = false, framePending = false, manualTheory = false;
let lastScrollY = window.scrollY, lastCheckpoint = -1;
let curveData = null;
const paused = () => userPaused || motionMedia.matches;

function selectButtons(selector, chosen) {
  $$(selector).forEach((button) => button.setAttribute('aria-pressed', String(button === chosen)));
}
function setTheoryStep(index) {
  $$('[data-theory-copy]').forEach((p) => { p.hidden = Number(p.dataset.theoryCopy) !== index; });
  selectButtons('[data-theory-step]', $('[data-theory-step="' + index + '"]'));
}
function progressOf(scene, fallback = 1) {
  const panel = scene.querySelector('.scene-panel');
  const style = window.getComputedStyle(panel);
  if (style.position !== 'sticky') return fallback;
  const stickyTop = parseFloat(style.top) || 0;
  return sceneProgress(scene.getBoundingClientRect().top, scene.offsetHeight, panel.offsetHeight, stickyTop);
}
function showComparison(progress) {
  const position = (1 - progress) * 100;
  $('#comparison-clip').setAttribute('x', String(position * 10));
  $('#comparison-clip').setAttribute('width', String(1000 - position * 10));
  $('#comparison-divider').style.left = position + '%';
}
function scrollFrame() {
  framePending = false;
  const y = window.scrollY;
  const total = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  $('#reading-progress').style.transform = 'scaleX(' + Math.min(1, Math.max(0, y / total)) + ')';
  if (!paused()) {
    const hero = $('#top');
    hero.style.setProperty('--hero-progress', Math.min(1, y / hero.offsetHeight));
    const theory = $('#theory');
    const extent = theory.offsetHeight - window.innerHeight;
    const progress = Math.max(0, Math.min(1, (y - theory.offsetTop) / Math.max(1, extent)));
    theory.style.setProperty('--theory-progress', progress);
    if (Math.abs(y - lastScrollY) > 3) manualTheory = false;
    if (!manualTheory && window.innerWidth > 680 && extent > 100) setTheoryStep(Math.min(2, Math.floor(progress * 3)));
  }
  showComparison(paused() ? .5 : progressOf($('#interaction'), .5));
  if (curveData) {
    const index = paused() ? curveData.series.points.length - 1 : scrollCheckpointIndex(curveData, progressOf($('#learning')));
    if (index !== lastCheckpoint) { showCheckpoint(curveData, index); lastCheckpoint = index; }
  }
  lastScrollY = y;
}
function requestFrame() {
  if (!framePending) { framePending = true; window.requestAnimationFrame(scrollFrame); }
}
function syncMotion() {
  document.documentElement.classList.toggle('motion-paused', paused());
  const button = $('#motion-toggle');
  button.disabled = motionMedia.matches;
  button.setAttribute('aria-pressed', String(paused()));
  const label = motionMedia.matches ? '系统已开启减少动态效果' : paused() ? '启用动态效果' : '暂停动态效果';
  button.setAttribute('aria-label', label); button.title = label;
  button.firstElementChild.textContent = paused() ? '▷' : 'Ⅱ';
  requestFrame();
}
$('#motion-toggle').addEventListener('click', () => { userPaused = !userPaused; syncMotion(); });
motionMedia.addEventListener('change', syncMotion);
window.addEventListener('scroll', requestFrame, {passive: true});
window.addEventListener('resize', requestFrame, {passive: true});
window.addEventListener('load', requestFrame, {once: true});
$$('[data-theory-step]').forEach((button) => button.addEventListener('click', () => {
  manualTheory = true; setTheoryStep(Number(button.dataset.theoryStep));
}));
document.documentElement.classList.add('scroll-enabled');
syncMotion();

if ('IntersectionObserver' in window) {
  document.documentElement.classList.add('js');
  const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
    if (entry.isIntersecting) { entry.target.classList.add('revealed'); observer.unobserve(entry.target); }
  }), {threshold: .08, rootMargin: '0px 0px -24px 0px'});
  $$('[data-reveal]').forEach((el) => observer.observe(el));
}

const elements = {
  mlp: ['逐点计算', 'MLP', '对每个 token 的内部表示做非线性变换。它提供局部计算，再与跨 token 的交互协同。', 'x → σ(Wx)', 'Code-Map', '阅读组件地图'],
  glu: ['门控激活', 'GLU', '两条变换路径逐元素相乘：一条携带信息，另一条控制信息如何通过。', 'f(x) ⊙ g(x)', 'Research-Program', '阅读研究路线'],
  colu: ['从对称性到设计', 'CoLU', '从对称性约束出发构造激活函数，用具体候选连接理论设计与小模型实验。', '对称性 → 候选结构', 'https://proceedings.mlr.press/v285/fu24a.html', '阅读 CoLU 论文'],
  linear: ['递归状态', '线性注意力', '把历史信息压缩进可更新的状态，再由查询读出。状态更新为长序列提供另一种计算路径。', 'S ← S + k ⊗ v', 'Research-Program', '阅读研究路线'],
  softmax: ['内容寻址', 'Softmax 注意力', '查询与每个键匹配，归一化得到权重，再汇总对应的值。它是普通对照组的基本交互。', 'softmax(qkᵀ) · v', 'Limite-SFT-Comparison', '阅读对照实验'],
  sparse: ['选择性交互', '稀疏注意力', '通过窗口、路由或其他选择机制，限制每次参与计算的历史位置，让交互集中在选定的信息上。', '关注一个子集', 'DeepSeek-Experiments', '阅读架构实验'],
  simplicial: ['三体交互', '2-单纯形注意力', '一个查询，同时对一对历史 token 评分。把两份信息的组合作为直接的交互单元。', 'q · (k ⊙ k′)', 'Limite-SFT-Comparison', '阅读实验设计'],
  triadic: ['探索中的候选', 'Triadic', '同一个历史 token 的两个键与一个值写入三阶递归状态，再由两个查询读出。它与 2-simplicial 组合两个历史位置的机制不同。', 'S ← S + k ⊗ k′ ⊗ v', 'https://arxiv.org/abs/2609.36529', '阅读 Triadic 论文'],
};
$$('[data-element]').forEach((button) => button.addEventListener('click', () => {
  const [kicker, name, description, formula, destination, linkText] = elements[button.dataset.element];
  selectButtons('[data-element]', button);
  $('#element-kicker').textContent = kicker;
  $('#element-name').textContent = name;
  $('#element-description').textContent = description;
  $('#element-formula').textContent = formula;
  $('#element-link').href = destination.startsWith('https://') ? destination : wiki + destination;
  $('#element-link').textContent = linkText + ' ↗';
}));

const svgNS = 'http://www.w3.org/2000/svg';
function svgElement(tag, attrs, text) {
  const node = document.createElementNS(svgNS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderCurve(data) {
  const grid = $('#chart-grid');
  grid.replaceChildren();
  [.80, .82, .84, .86].forEach((value) => {
    const [, y] = chartPoint(CHART.xmin, value);
    grid.append(svgElement('line', {x1: CHART.left, x2: CHART.right, y1: y, y2: y}));
    grid.append(svgElement('text', {x: CHART.left - 20, y: y + 4, 'text-anchor': 'end'}, value.toFixed(2)));
  });
  [2, 4, 6, 8, 10].forEach((value) => {
    const [x] = chartPoint(value * 1e9, .8);
    grid.append(svgElement('text', {x, y: 430, 'text-anchor': 'middle'}, value + 'B'));
  });
  $('#chart-paths').replaceChildren(
    svgElement('path', {id: 'normal-line', d: curvePath(data.series.points, 1)}),
    svgElement('path', {id: 'simplicial-line', d: curvePath(data.series.points, 2)}),
  );
}

function showCheckpoint(data, index) {
  const point = checkpointAt(data, index);
  const efficiency = checkpointEfficiency(data, index);
  const [x, yn] = chartPoint(point.tokens, point.normal);
  const [, ys] = chartPoint(point.tokens, point.simplicial);
  const [matchX, matchY] = chartPoint(efficiency.simplicialTokens, efficiency.simplicialLoss);
  $('#cursor-line').setAttribute('x1', x);
  $('#cursor-line').setAttribute('x2', x);
  $('#normal-cursor').setAttribute('cx', x);
  $('#normal-cursor').setAttribute('cy', yn);
  $('#simplicial-cursor').setAttribute('cx', x);
  $('#simplicial-cursor').setAttribute('cy', ys);
  $('#token-readout').textContent = (point.tokens / 1e9).toFixed(2) + 'B';
  $('#normal-readout').textContent = point.normal.toFixed(6);
  $('#simplicial-readout').textContent = point.simplicial.toFixed(6);
  $('#checkpoint-ratio').textContent = efficiency.ratio.toFixed(2) + '×';
  $('#checkpoint-budgets').textContent = (efficiency.normalTokens / 1e9).toFixed(2) + 'B ÷ '
    + (efficiency.simplicialTokens / 1e9).toFixed(2) + 'B';
  $('#efficiency-target-line').setAttribute('x1', matchX);
  $('#efficiency-target-line').setAttribute('x2', x);
  $('#efficiency-target-line').setAttribute('y1', yn);
  $('#efficiency-target-line').setAttribute('y2', yn);
  $('#efficiency-crossing-stem').setAttribute('x1', matchX);
  $('#efficiency-crossing-stem').setAttribute('x2', matchX);
  $('#efficiency-crossing-stem').setAttribute('y1', yn);
  $('#efficiency-crossing-stem').setAttribute('y2', matchY);
  $('#efficiency-crossing').setAttribute('cx', matchX);
  $('#efficiency-crossing').setAttribute('cy', matchY);
}
function showTarget(data, key) {
  const result = targetComparison(data, key);
  const format = (v) => (v / 1e9).toFixed(2) + 'B';
  $('#efficiency-ratio').textContent = result.ratio.toFixed(2);
  $('#normal-budget').textContent = format(result.normalTokens);
  $('#simplicial-budget').textContent = format(result.simplicialTokens);
  $('#normal-budget-bar').style.width = result.normalTokens / 1e8 + '%';
  $('#simplicial-budget-bar').style.width = result.simplicialTokens / 1e8 + '%';
  $('#token-saving').textContent = result.savingPercent.toFixed(1) + '%';
  const endpoint = result.mode === 'fixed_endpoint_reference';
  $('#target-summary').textContent = endpoint
    ? '达到普通组 10B 终点的验证 CE，三体首次保存达标点约为 ' + format(result.simplicialTokens) + '。'
    : '达到 CE ≤ ' + result.target.toFixed(3) + '，比较两组首次达标的实际保存点。';
  $('#target-note').textContent = endpoint
    ? '10B 为普通组终点参照；三体取首次达到该 CE 的实际保存点。未作插值。'
    : '两组均取首次观测到 CE 不高于目标的实际保存点。保存间隔内不作插值。';
  selectButtons('[data-target]', $('[data-target="' + key + '"]'));
}


async function loadEvidence() {
  try {
    const [data, warmup] = await Promise.all(['learning-curves.json', 'warmup.json'].map(async (name) => {
      const response = await fetch(new URL('./assets/' + name, import.meta.url));
      if (!response.ok) throw new Error('Evidence file unavailable');
      return response.json();
    }));
    validateEvidence(data, warmup);
    renderCurve(data);
    curveData = data;
    $$('[data-target]').forEach((button) => button.addEventListener('click', () => showTarget(data, button.dataset.target)));
    showTarget(data, 'endpoint');
    document.documentElement.dataset.evidence = 'ready';
    requestFrame();
  } catch {
    $('#data-status').textContent = '当前显示已核实的静态图表；完整记录可在实验文档中查看。';
    document.documentElement.dataset.evidence = 'unavailable';
  }
}
loadEvidence();
