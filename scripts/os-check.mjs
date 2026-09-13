#!/usr/bin/env node
// 开源解读的质量检查(慢检查,单独跑):
//   1) 每份 diagrams/*.json 用 archify 做 showcase 校验,要求 ok 且零错零警;architecture 附带 --repo-root 核对来源
//   2) 每份规格都有 public/opensource/<项目>/<slug>.svg 与 .html,且产物不比规格旧
//   3) 每页 markdown 引用的 /opensource/<项目>/... 文件都存在
//   4) 每份 _NN-evidence.md 的「位置」与「原样引用」逐条拿到 projects/<主题>/<项目>/ 的源码上核对
//   5) 开跑前先把被检查项目的源码仓同步到上游最新(见 scripts/projects-sync.mjs),
//      再核对解读页记录的源码基准 commit 是否就是当前 HEAD;漂移且动到被引用文件才判失败,
//      未触及引用点只提醒(上游高频合并时绝大多数漂移与本解读无关)
// 用法: node scripts/os-check.mjs [--project <项目名>] [--skip-archify] [--skip-sync]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { syncProjects } from './projects-sync.mjs';

const ROOT = process.cwd();
const args = process.argv.slice(2);
const onlyProject = args.includes('--project') ? args[args.indexOf('--project') + 1] : null;
const skipArchify = args.includes('--skip-archify');
const skipSync = args.includes('--skip-sync') || process.env.OS_CHECK_SKIP_SYNC === '1';
const ARCHIFY = path.join(process.env.ARCHIFY_HOME || path.join(os.homedir(), '.claude', 'skills', 'archify'), 'bin', 'archify.mjs');

let failures = 0;
const fail = (msg) => { failures += 1; console.log('  ✗ ' + msg); };
const ok = (msg) => console.log('  ✓ ' + msg);
let warnings = 0;
const warn = (msg) => { warnings += 1; console.log('  ! ' + msg); };

function listProjects() {
  const out = [];
  const osRoot = path.join(ROOT, 'opensource');
  for (const topic of fs.readdirSync(osRoot, { withFileTypes: true })) {
    if (!topic.isDirectory() || topic.name.startsWith('.') || topic.name.startsWith('_')) continue;
    for (const proj of fs.readdirSync(path.join(osRoot, topic.name), { withFileTypes: true })) {
      if (!proj.isDirectory() || proj.name.startsWith('.') || proj.name.startsWith('_')) continue;
      if (onlyProject && proj.name !== onlyProject) continue;
      out.push({ topic: topic.name, project: proj.name, dir: path.join(osRoot, topic.name, proj.name) });
    }
  }
  return out;
}

function checkDiagrams({ topic, project, dir }) {
  const dDir = path.join(dir, 'diagrams');
  if (!fs.existsSync(dDir)) return;
  const pub = path.join(ROOT, 'public', 'opensource', project);
  const repoRoot = path.join(ROOT, 'projects', topic, project);
  for (const f of fs.readdirSync(dDir).filter((x) => x.endsWith('.json') && !x.startsWith('_')).sort()) {
    const spec = path.join(dDir, f);
    const slug = f.replace(/\.json$/, '');
    const svg = path.join(pub, slug + '.svg');
    const html = path.join(pub, slug + '.html');
    for (const [p, label] of [[svg, 'svg'], [html, 'html']]) {
      if (!fs.existsSync(p)) { fail(`${slug}: 缺产物 ${label}`); continue; }
      if (fs.statSync(p).mtimeMs < fs.statSync(spec).mtimeMs) fail(`${slug}: ${label} 比规格旧,需重新交付导出`);
    }
    if (skipArchify) continue;
    let type;
    try { type = JSON.parse(fs.readFileSync(spec, 'utf8')).diagram_type; } catch (e) { fail(`${slug}: 规格不是合法 JSON(${e.message})`); continue; }
    const cmd = [ARCHIFY, 'validate', type, spec, '--quality', 'showcase', '--json'];
    if (type === 'architecture' && fs.existsSync(repoRoot)) cmd.push('--repo-root', repoRoot);
    let r;
    try {
      r = JSON.parse(execFileSync('node', cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 16 * 1024 * 1024 }));
    } catch (e) {
      try { r = JSON.parse(e.stdout || ''); } catch { fail(`${slug}: archify 校验无法运行(${(e.message || '').split('\n')[0]})`); continue; }
    }
    const sum = r.composition?.summary || {};
    if (r.ok && (sum.errors ?? 0) === 0 && (sum.warnings ?? 0) === 0) ok(`${slug} (${type}) showcase 零错零警${r.evidence?.verified ? `,来源核对 ${r.evidence.references} 处` : ''}`);
    else fail(`${slug} (${type}) 校验未过:${(r.error || '').split('\n')[0] || JSON.stringify(sum)}`);
  }
}

function checkPageAssets({ project, dir }) {
  const pub = path.join(ROOT, 'public', 'opensource', project);
  let refs = 0;
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.md') && !x.startsWith('_') && !x.startsWith('.'))) {
    const md = fs.readFileSync(path.join(dir, f), 'utf8');
    for (const m of md.matchAll(/\]\((\/opensource\/([^/)]+)\/([^)]+))\)/g)) {
      refs += 1;
      if (m[2] !== project) fail(`${f}: 引用了别的项目的产物 ${m[1]}`);
      else if (!fs.existsSync(path.join(pub, m[3]))) fail(`${f}: 引用的 ${m[1]} 不存在`);
    }
  }
  if (refs) ok(`页面引用的 ${refs} 个产物文件都存在`);
}

// ---- 导出 SVG 里的文字互相遮挡 ----
// archify 给每段文字配了一块精确的白底板(class="c-mask"),文字就画在它上面。
// 两块底板一旦相交,后画的那块会把先画的文字盖掉一截——这是纯粹的 bug,没有正常情况。
// archify 自己的 showcase 校验比的是文字本身的范围,底板比文字宽,所以它拦不住这一类(实测漏过两次)。
function labelPlates(svg) {
  const out = [];
  for (const m of svg.matchAll(/<rect\b[^>]*class="c-mask"[^>]*?\/?>/g)) {
    const a = Object.fromEntries([...m[0].matchAll(/([\w-]+)="([^"]*)"/g)].map((x) => [x[1], x[2]]));
    if (!a.x || !a.width) continue;
    // 只认「底板后面紧跟着 <text>」的那种,中间夹了别的 <rect> 的是节点或容器的底板,不算
    const tail = svg.slice(m.index + m[0].length, m.index + m[0].length + 240);
    const nextText = tail.indexOf('<text');
    const nextRect = tail.indexOf('<rect');
    if (nextText < 0 || (nextRect >= 0 && nextRect < nextText)) continue;
    const t = tail.slice(nextText).match(/<text[^>]*>([^<]{1,60})/);
    if (!t) continue;
    out.push({ txt: t[1].trim(), x: +a.x, y: +a.y, w: +a.width, h: +a.height });
  }
  return out;
}

function checkLabelOverlap({ project }) {
  const pub = path.join(ROOT, 'public', 'opensource', project);
  if (!fs.existsSync(pub)) return;
  const svgs = fs.readdirSync(pub).filter((x) => x.endsWith('.svg')).sort();
  if (!svgs.length) return;
  let hits = 0;
  for (const f of svgs) {
    const plates = labelPlates(fs.readFileSync(path.join(pub, f), 'utf8'));
    for (let i = 0; i < plates.length; i += 1) {
      for (let j = i + 1; j < plates.length; j += 1) {
        const a = plates[i];
        const b = plates[j];
        const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
        const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
        if (ox > 1 && oy > 1) {
          hits += 1;
          fail(`${f}: 「${a.txt}」与「${b.txt}」的文字底板重叠 ${ox.toFixed(0)}×${oy.toFixed(0)}px,有一段字会被盖住`);
        }
      }
    }
  }
  if (!hits) ok(`${svgs.length} 张导出图的文字没有互相遮挡`);
}

// ---- 证据表核对 ----
const norm = (s) => s.replace(/\\\|/g, '|').replace(/\s+/g, ' ').trim();
const squash = (s) => norm(s).replace(/\s+/g, '');
function resolvePath(repoRoot, p) {
  const cands = [p, `nanovllm/${p}`, `nanovllm/engine/${p}`, `nanovllm/layers/${p}`, `nanovllm/utils/${p}`, `nanovllm/models/${p}`];
  for (const c of cands) { const full = path.join(repoRoot, c); if (fs.existsSync(full) && fs.statSync(full).isFile()) return full; }
  return null;
}
const fileCache = new Map();
const readLines = (full) => { if (!fileCache.has(full)) fileCache.set(full, fs.readFileSync(full, 'utf8').split('\n')); return fileCache.get(full); };
const RANGE_TAIL = String.raw`(\d+)(?:-(\d+))?((?:\s*[,、/]\s*\d+(?:-\d+)?)*)`;
const REF_RE = new RegExp('`([\\w./-]+\\.py)(?::' + RANGE_TAIL + ')?`|`:' + RANGE_TAIL + '`', 'g');
const parseRanges = (a, b, tail) => {
  const ranges = [[+a, +(b || a)]];
  for (const c of (tail || '').matchAll(/(\d+)(?:-(\d+))?/g)) ranges.push([+c[1], +(c[2] || c[1])]);
  return ranges;
};
function parseRefs(rowText, repoRoot) {
  // `file.py:12-34, 40` 正常引用;`:51-52` 是续写,沿用本行或本表上一次出现的文件
  const refs = []; let last = null; // 裸 `:12-34` 只沿用同一行里前一个文件;行首必须写文件名
  for (const m of rowText.matchAll(REF_RE)) {
    if (m[1]) {
      const full = resolvePath(repoRoot, m[1]);
      if (!full) { refs.push({ file: m[1], missing: true }); last = null; continue; }
      const ranges = m[2] ? parseRanges(m[2], m[3], m[4]) : [[1, readLines(full).length]];
      last = { file: m[1], full, ranges }; refs.push(last);
    } else if (!last) {
      refs.push({ file: '(行首裸行号 `' + m[0] + '` 没有文件名)', missing: true });
    } else if (last) {
      if (!refs.includes(last)) { last = { file: last.file, full: last.full, ranges: [] }; refs.push(last); }
      last.ranges.push(...parseRanges(m[5], m[6], m[7]));
    }
  }
  return refs;
}
const isRefLike = (raw) => /^[\w./-]+\.py(:[\d,、/ -]*)?$/.test(raw) || /^:[\d,、/ -]+$/.test(raw);

function checkEvidence({ topic, project, dir }) {
  const repoRoot = path.join(ROOT, 'projects', topic, project);
  if (!fs.existsSync(repoRoot)) { console.log(`  · projects/${topic}/${project} 不存在,跳过证据表核对`); return; }
  for (const f of fs.readdirSync(dir).filter((x) => /^_\d+.*evidence\.md$/.test(x)).sort()) {
    const lines = fs.readFileSync(path.join(dir, f), 'utf8').split('\n');
    let rows = 0, snippets = 0; const misses = [];
    let quoteCol = -1; // 当前表格里「原样引用」列的下标,-1 表示查所有单元格
    lines.forEach((line, i) => {
      if (!line.startsWith('|')) return;
      const cells = line.split(/(?<!\\)\|/).slice(1, -1);
      if (/^\|\s*-{2,}/.test(line)) return;
      const next = lines[i + 1] || '';
      if (/^\|\s*:?-{2,}/.test(next)) { // 这是表头行
        quoteCol = cells.findIndex((c) => /原样引用|原样|引用|代码/.test(c));
        return;
      }
      if (cells.length < 2) return;
      const refs = parseRefs(line, repoRoot);
      if (!refs.length) return;
      rows += 1;
      for (const r of refs.filter((x) => x.missing)) misses.push(`第 ${i + 1} 行:找不到文件 ${r.file}`);
      const good = refs.filter((x) => !x.missing);
      if (!good.length) return;
      const hay = squash(good.map((r) => r.ranges.map(([a, b]) => readLines(r.full).slice(Math.max(0, a - 4), b + 3).join(' ')).join(' ')).join(' '));
      const targets = quoteCol >= 0 && quoteCol < cells.length ? [cells[quoteCol]] : cells;
      for (const c of targets) {
        if (/^\s*干跑/.test(c)) continue; // 「干跑输出 `...`」是轨迹记录,不是源码引用
        for (const s of c.matchAll(/`([^`]+)`/g)) {
          const raw = s[1];
          if (/干跑输出\s*$/.test(c.slice(Math.max(0, s.index - 8), s.index))) continue; // 干跑输出不是源码引用
          if (isRefLike(raw)) continue;
          if (raw.length < 8 || !/[=().\[_:]/.test(raw)) continue;
          snippets += 1;
          const parts = norm(raw).split(/\s*(?:…|\.\.\.|\s\/\s)\s*/).map(squash).filter((p) => p.length >= 6);
          if (!parts.every((p) => hay.includes(p))) misses.push(`第 ${i + 1} 行:「${raw.slice(0, 70)}」不在所引行号 ±3 行内`);
        }
      }
    });
    if (misses.length) { fail(`${f}:${rows} 行、${snippets} 段引用,${misses.length} 处未命中`); misses.forEach((m) => console.log('      ' + m)); }
    else ok(`${f}:${rows} 行、${snippets} 段原样引用全部命中源码`);
  }
}

// ---- 源码基准核对 ----
// 解读的每一条断言都钉在一个 commit 上。工作树被同步到上游最新之后,
// 若解读页记录的基准不是当前 HEAD,这份解读就已经过期,必须按 06 的「已有解读更新」复核。
function citedSourceFiles(dir) {
  // 底稿与证据表里所有形如 `path/to/file.py` 或 `path/to/file.py:12-34` 的引用
  const files = new Set();
  for (const f of fs.readdirSync(dir).filter((x) => x.startsWith('_') && x.endsWith('.md'))) {
    const md = fs.readFileSync(path.join(dir, f), 'utf8');
    for (const m of md.matchAll(/`([\w./-]+\.(?:py|cu|cuh|h|cpp|hpp|rs|mjs|ts))(?::[\d,、/ -]*)?`/g)) files.add(m[1]);
  }
  return files;
}

function checkBaseline({ topic, project, dir }) {
  const repoRoot = path.join(ROOT, 'projects', topic, project);
  if (!fs.existsSync(repoRoot)) { console.log(`  · projects/${topic}/${project} 不存在,跳过基准核对`); return; }
  const overview = fs.readdirSync(dir).filter((f) => /^00-.*\.md$/.test(f))[0];
  if (!overview) { warn(`没有 00- 总览页,无法核对源码基准`); return; }
  const md = fs.readFileSync(path.join(dir, overview), 'utf8');
  const m = md.match(/commit\s+`([0-9a-f]{7,40})`/);
  if (!m) { warn(`${overview}: 没有记录源码基准 commit(格式: commit \`<sha>\`),无法判断解读是否已过期`); return; }
  const git = (a) => execFileSync('git', ['-C', repoRoot, ...a], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }).trim();
  let head;
  try { head = git(['rev-parse', 'HEAD']); }
  catch (e) { fail(`读不到 projects/${topic}/${project} 的 HEAD(${(e.message || '').split('\n')[0]})`); return; }
  const recorded = m[1];
  if (head.startsWith(recorded)) { ok(`源码基准 ${recorded.slice(0, 12)} 与工作树 HEAD 一致`); return; }

  // 基准漂移了。只有当漂移真的动到被引用的文件时才算解读过期;
  // 上游高频合并(vLLM 一天几十个提交)时,绝大多数漂移与本解读无关。
  let changed, count;
  try {
    changed = new Set(git(['diff', '--name-only', `${recorded}..HEAD`]).split('\n').filter(Boolean));
    count = git(['rev-list', '--count', `${recorded}..HEAD`]);
  } catch (e) {
    fail(`${overview}: 基准 ${recorded.slice(0, 12)} 在工作树里找不到(${(e.message || '').split('\n')[0]});请重新核对并更新基准`);
    return;
  }
  const cited = citedSourceFiles(dir);
  const hit = [...cited].filter((f) => changed.has(f)).sort();
  if (!hit.length) {
    warn(
      `基准漂移 ${count} 个提交(${recorded.slice(0, 12)} → ${head.slice(0, 12)}),` +
        `但未触及本解读引用的任何文件(共引用 ${cited.size} 个);把 ${overview} 的基准更新为 ${head.slice(0, 12)} 即可`,
    );
    return;
  }
  fail(
    `源码基准已过期:${overview} 记录 ${recorded.slice(0, 12)},工作树 HEAD 是 ${head.slice(0, 12)}(相差 ${count} 个提交),` +
      `其中 ${hit.length} 个被引用的文件有改动,需按 06 的「已有解读更新」逐条复核:${hit.slice(0, 8).join('、')}${hit.length > 8 ? ' 等' : ''}`,
  );
}

const projects = listProjects();

if (!skipSync) {
  console.log('== 同步 projects/ 下被检查的源码仓到上游最新');
  const names = new Set(projects.map((p) => p.project));
  const results = syncProjects({ filter: names, log: (m) => console.log(m) });
  for (const r of results) {
    if (['dirty', 'ahead', 'detached', 'no-upstream', 'fetch-failed', 'error'].includes(r.state)) {
      fail(`${r.rel} 无法同步到最新:${r.detail}`);
    }
  }
} else {
  console.log('== 跳过源码仓同步(--skip-sync)');
}

for (const p of projects) {
  console.log(`\n== ${p.topic}/${p.project}`);
  checkBaseline(p);
  checkDiagrams(p);
  checkPageAssets(p);
  checkLabelOverlap(p);
  checkEvidence(p);
}
console.log(failures ? `\n共 ${failures} 项未过${warnings ? `、${warnings} 项提醒` : ''}` : `\n全部通过${warnings ? `(${warnings} 项提醒)` : ''}`);
process.exit(failures ? 1 : 0);
