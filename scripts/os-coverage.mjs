#!/usr/bin/env node
// 拆章后的覆盖率核对:哪些够体量的源码目录还没有任何一章认领。
//
// 06「主线程 → ② 覆盖率核对」说「逐个目录问它归哪一章……不能靠印象」——这就是那个「不靠印象」的办法。
// 依据是代码索引页 99-代码索引.md:每一章在那里登记了自己读过哪些文件,反过来就能算出谁没人管。
//
// 用法: node scripts/os-coverage.mjs <解读目录> <源码根> [--min-kb=120]
//   例: node scripts/os-coverage.mjs "opensource/推理服务/sglang" /tmp/sgl-base/python/sglang/srt
import fs from 'node:fs';
import path from 'node:path';

const [readDir, srcRoot] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const minKb = Number((process.argv.find((a) => a.startsWith('--min-kb=')) || '--min-kb=120').split('=')[1]);
if (!readDir || !srcRoot) {
  console.error('用法: node scripts/os-coverage.mjs <解读目录> <源码根> [--min-kb=120]');
  process.exit(2);
}

/** 代码索引页里每一章登记了哪些文件路径 */
function claimsByChapter(indexFile) {
  const out = new Map();
  let cur = null;
  for (const line of fs.readFileSync(indexFile, 'utf8').split('\n')) {
    const h = line.match(/^## (\d\d)\|/);
    if (h) { cur = h[1]; out.set(cur, new Set()); continue; }
    if (!cur) continue;
    for (const m of line.matchAll(/([\w./-]+\.(?:py|rs|mjs|ts|cpp|cu|h))/g)) out.get(cur).add(m[1]);
  }
  return out;
}

/** 源码树里每个目录自己的体量(不含子目录),只数代码文件 */
function dirSizes(root) {
  const sizes = new Map();
  const walk = (dir) => {
    let own = 0;
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) { if (!/^(__pycache__|\.|tests?$|benchmarks?$)/.test(e.name)) walk(p); }
      else if (/\.(py|rs|cpp|cu|h)$/.test(e.name)) own += fs.statSync(p).size;
    }
    sizes.set(path.relative(root, dir) || '.', own);
  };
  walk(root);
  return sizes;
}

const index = path.join(readDir, '99-代码索引.md');
if (!fs.existsSync(index)) { console.error(`找不到 ${index}`); process.exit(2); }
const claims = claimsByChapter(index);
const sizes = [...dirSizes(srcRoot)].filter(([, s]) => s > 0).sort((a, b) => b[1] - a[1]);

/** 一条登记路径落在哪个目录:拿目录名去和登记路径做前缀/包含匹配 */
const covers = (dir, claimed) => (dir === '.' ? !claimed.includes('/') : claimed.includes(`${dir}/`));

let uncovered = 0;
console.log(`源码根 ${srcRoot}`);
console.log(`阈值 ${minKb} KB —— 超过它又没人认领的,要么补进章表,要么在底稿写明为什么不写\n`);
console.log('体量(KB)  目录                                          认领的章');
for (const [dir, bytes] of sizes) {
  const kb = bytes / 1024;
  const who = [...claims.entries()].filter(([, set]) => [...set].some((c) => covers(dir, c))).map(([n]) => n);
  const big = kb >= minKb;
  if (!who.length && !big) continue;
  if (!who.length && big) uncovered += 1;
  const mark = !who.length && big ? ' ✗ ' : '   ';
  console.log(`${mark}${kb.toFixed(0).padStart(6)}  ${dir.padEnd(45)} ${who.length ? who.join(' ') : '—— 没人认领'}`);
}
console.log(uncovered ? `\n✗ ${uncovered} 个够体量的目录没人认领` : `\n✓ 够体量的目录都有章认领`);
process.exit(uncovered ? 1 : 0);
