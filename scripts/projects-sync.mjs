#!/usr/bin/env node
// 把 projects/ 下的源码仓库拉到各自上游的最新提交。
//   · 自动发现:凡 projects/ 下带 .git 的目录都算,新增项目无需登记
//   · 只做快进:工作树干净且只落后上游时 git merge --ff-only,绝不 merge、rebase、reset
//   · 不安全的一律不碰并报告:有未提交改动、有本地提交、游离 HEAD、无上游、fetch 失败
// 用法: node scripts/projects-sync.mjs [--dry-run] [--only <项目名>[,<项目名>…]] [--quiet]
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const ROOT = process.cwd();
const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const quiet = args.includes('--quiet');
const onlyArg = args.includes('--only') ? args[args.indexOf('--only') + 1] : null;
const only = onlyArg ? new Set(onlyArg.split(',').map((s) => s.trim()).filter(Boolean)) : null;

const FETCH_TIMEOUT_MS = 120_000;

export function listProjectRepos(root = ROOT) {
  const base = path.join(root, 'projects');
  if (!fs.existsSync(base)) return [];
  const out = [];
  // 主题目录 → 项目目录;两层足够,同时兼容 projects/<项目> 这种没有主题层的放法
  const walk = (dir, depth) => {
    if (depth > 2) return;
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!e.isDirectory() || e.name.startsWith('.')) continue;
      const full = path.join(dir, e.name);
      if (fs.existsSync(path.join(full, '.git'))) out.push(full);
      else walk(full, depth + 1);
    }
  };
  walk(base, 0);
  return out.sort();
}

const git = (dir, cmdArgs, timeout) =>
  execFileSync('git', ['-C', dir, ...cmdArgs], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    timeout,
    // 快进几千个提交时 git 的 diffstat 很长,默认 1MB 缓冲会 ENOBUFS
    maxBuffer: 256 * 1024 * 1024,
  }).trim();

// 返回 { state, detail, … }。state: latest | advanced | dirty | ahead | detached | no-upstream | fetch-failed | error
export function inspectRepo(dir, { fetch = true, apply = true } = {}) {
  const rel = path.relative(ROOT, dir);
  const info = { dir, rel, name: path.basename(dir) };
  try {
    const branch = git(dir, ['rev-parse', '--abbrev-ref', 'HEAD']);
    info.branch = branch;
    if (branch === 'HEAD') return { ...info, state: 'detached', detail: '游离 HEAD,未在任何分支上' };

    let upstream = null;
    try {
      upstream = git(dir, ['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}']);
    } catch {
      return { ...info, state: 'no-upstream', detail: `分支 ${branch} 没有设置上游` };
    }
    info.upstream = upstream;
    const remote = upstream.split('/')[0];

    if (fetch) {
      try {
        git(dir, ['fetch', '--quiet', '--prune', remote], FETCH_TIMEOUT_MS);
      } catch (first) {
        // 远端删过分支又重建同名 tag 之类会让 --prune 报 incorrect old value;强制覆盖本地远端跟踪引用再试一次
        try {
          git(dir, ['fetch', '--quiet', '--prune', '--prune-tags', '--force', remote], FETCH_TIMEOUT_MS);
        } catch (second) {
          const msg =
            (second.stderr || second.message || first.stderr || first.message || '')
              .split('\n')
              .filter(Boolean)
              .pop() || '未知错误';
          return { ...info, state: 'fetch-failed', detail: msg.slice(0, 160) };
        }
      }
    }

    const dirty = git(dir, ['status', '--porcelain']).split('\n').filter(Boolean).length;
    const [ahead, behind] = git(dir, ['rev-list', '--left-right', '--count', `HEAD...${upstream}`])
      .split(/\s+/)
      .map(Number);
    Object.assign(info, { dirty, ahead, behind });

    if (ahead > 0) {
      return {
        ...info,
        state: 'ahead',
        detail: `本地有 ${ahead} 个自己的提交${behind ? `,同时落后上游 ${behind} 个` : ''};不擅自 rebase`,
      };
    }
    if (behind === 0) return { ...info, state: 'latest', detail: '已是上游最新' };
    if (dirty > 0) {
      return { ...info, state: 'dirty', detail: `落后上游 ${behind} 个提交,但有 ${dirty} 处未提交改动` };
    }
    if (!apply || dryRun) {
      return { ...info, state: 'advanced', applied: false, detail: `可快进 ${behind} 个提交(dry-run,未执行)` };
    }
    const before = git(dir, ['rev-parse', '--short', 'HEAD']);
    git(dir, ['merge', '--ff-only', '--quiet', '--no-stat', upstream]);
    const after = git(dir, ['rev-parse', '--short', 'HEAD']);
    return { ...info, state: 'advanced', applied: true, before, after, detail: `快进 ${behind} 个提交 ${before} → ${after}` };
  } catch (e) {
    return { ...info, state: 'error', detail: (e.message || '').split('\n')[0].slice(0, 160) };
  }
}

const ICON = {
  latest: '·',
  advanced: '↑',
  dirty: '✗',
  ahead: '✗',
  detached: '✗',
  'no-upstream': '✗',
  'fetch-failed': '✗',
  error: '✗',
};
const BLOCKED = new Set(['dirty', 'ahead', 'detached', 'no-upstream', 'fetch-failed', 'error']);

export function syncProjects({ apply = true, fetch = true, filter = null, log = console.log } = {}) {
  const repos = listProjectRepos().filter((d) => !filter || filter.has(path.basename(d)));
  const results = [];
  for (const dir of repos) {
    const r = inspectRepo(dir, { fetch, apply });
    results.push(r);
    if (!quiet || BLOCKED.has(r.state)) log(`  ${ICON[r.state]} ${r.rel}: ${r.detail}`);
  }
  return results;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  console.log(`把 projects/ 下的仓库同步到各自上游最新${dryRun ? '(dry-run)' : ''}\n`);
  const results = syncProjects({ apply: !dryRun, filter: only });
  const blocked = results.filter((r) => BLOCKED.has(r.state));
  const moved = results.filter((r) => r.state === 'advanced' && r.applied);
  console.log(
    `\n共 ${results.length} 个仓库:已最新 ${results.filter((r) => r.state === 'latest').length}` +
      `,本次快进 ${moved.length},需人工处理 ${blocked.length}`,
  );
  if (moved.length) {
    console.log('\n已前进的仓库若已有解读,请按 06 的「已有解读更新」流程复核基准与证据表:');
    for (const r of moved) console.log(`  · ${r.rel} ${r.before} → ${r.after}`);
  }
  if (blocked.length) {
    console.log('\n需人工处理(脚本按守则不擅自改写这些仓库):');
    for (const r of blocked) console.log(`  · ${r.rel} — ${r.detail}`);
  }
  process.exit(blocked.length ? 1 : 0);
}
