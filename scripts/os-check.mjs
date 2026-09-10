#!/usr/bin/env node
// 开源解读的质量检查(慢检查,单独跑):
//   1) 每份 diagrams/*.json 用 archify 做 showcase 校验,要求 ok 且零错零警;architecture 附带 --repo-root 核对来源
//   2) 每份规格都有 public/opensource/<项目>/<slug>.svg 与 .html,且产物不比规格旧
//   3) 每页 markdown 引用的 /opensource/<项目>/... 文件都存在
//   4) 每份 _NN-evidence.md 的「位置」与「原样引用」逐条拿到 projects/<主题>/<项目>/ 的源码上核对
// 用法: node scripts/os-check.mjs [--project <项目名>] [--skip-archify]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const ROOT = process.cwd();
const args = process.argv.slice(2);
const onlyProject = args.includes('--project') ? args[args.indexOf('--project') + 1] : null;
const skipArchify = args.includes('--skip-archify');
const ARCHIFY = path.join(process.env.ARCHIFY_HOME || path.join(os.homedir(), '.claude', 'skills', 'archify'), 'bin', 'archify.mjs');

let failures = 0;
const fail = (msg) => { failures += 1; console.log('  ✗ ' + msg); };
const ok = (msg) => console.log('  ✓ ' + msg);

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

for (const p of listProjects()) {
  console.log(`\n== ${p.topic}/${p.project}`);
  checkDiagrams(p);
  checkPageAssets(p);
  checkEvidence(p);
}
console.log(failures ? `\n共 ${failures} 项未过` : '\n全部通过');
process.exit(failures ? 1 : 0);
