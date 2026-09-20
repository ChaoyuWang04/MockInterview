#!/usr/bin/env node
// 手写 SVG 的三类碰撞检查:文字越界、文字互压、文字被后画的图形盖住。
//
// 为什么要它:archify 给每段文字配白底板(class="c-mask"),os-check 原来靠底板相交来判遮挡;
// 但全库的图早已全部改成手写,手写图没有底板,那个检查会「找不到底板 → 直接报成功」,是假绿灯。
// 这里不依赖底板,直接按字号估算文字范围,浏览器不参与,谁都能跑。
//
// 用法: node scripts/svg-check.mjs <文件或目录…>
import fs from 'node:fs';
import path from 'node:path';

// 按字号估宽:CJK 与全角标点占满一格,ASCII 字母数字约 0.55,空格 0.28
const charW = (ch, size) => {
  const c = ch.codePointAt(0);
  if (ch === ' ') return size * 0.28;
  if (c > 0x2e7f) return size;            // CJK、全角标点、方块符号
  if (/[iIlj.,:;'|!]/.test(ch)) return size * 0.3;
  if (/[A-Z@%]/.test(ch)) return size * 0.65;
  return size * 0.55;
};
const textWidth = (s, size) => [...s].reduce((w, ch) => w + charW(ch, size), 0);

const attrs = (tag) => Object.fromEntries([...tag.matchAll(/([\w:-]+)="([^"]*)"/g)].map((m) => [m[1], m[2]]));
const num = (v, d = 0) => (v === undefined ? d : parseFloat(v));

/** 按文档顺序取出所有可见图元;文字额外算出估算 bbox */
function parse(svg) {
  const vb = (svg.match(/viewBox="([^"]+)"/) || [])[1];
  const [, , W, H] = vb ? vb.split(/[\s,]+/).map(Number) : [0, 0, 0, 0];
  const items = [];
  for (const m of svg.matchAll(/<(text|rect|polyline|line|circle|path)\b([^>]*?)(?:\/>|>([\s\S]*?)<\/\1>)/g)) {
    const [kind, a, inner] = [m[1], attrs(m[2]), m[3]];
    const at = m.index;
    if (kind === 'text') {
      const s = (inner || '').replace(/<[^>]*>/g, '').trim();
      if (!s) continue;
      const size = num(a['font-size'], 12);
      const w = textWidth(s, size);
      const anchor = a['text-anchor'] || 'start';
      const x = num(a.x) - (anchor === 'middle' ? w / 2 : anchor === 'end' ? w : 0);
      // 基线在 y,字面大致占 y-0.78size .. y+0.22size
      items.push({ kind, at, s, x, y: num(a.y) - size * 0.78, w, h: size });
    } else if (kind === 'rect') {
      if ((a.fill || '') === 'none') continue;   // 只描边的框不挡字
      items.push({ kind, at, x: num(a.x), y: num(a.y), w: num(a.width), h: num(a.height), fill: a.fill });
    } else if (kind === 'circle') {
      const [cx, cy, r] = [num(a.cx), num(a.cy), num(a.r)];
      items.push({ kind, at, x: cx - r, y: cy - r, w: r * 2, h: r * 2, fill: a.fill });
    }
    // polyline / line / path 是细描边,压字的情况靠人看,不进自动判定(误报太多)
  }
  return { W, H, items };
}

const hit = (a, b, pad = 1) =>
  Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x) > pad &&
  Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y) > pad;

export function checkFile(file) {
  const { W, H, items } = parse(fs.readFileSync(file, 'utf8'));
  const texts = items.filter((i) => i.kind === 'text');
  const shapes = items.filter((i) => i.kind !== 'text');
  const out = [];
  if (!W || !H) out.push('没有 viewBox,量不了');

  for (const t of texts) {
    if (t.x < -2 || t.y < -2 || t.x + t.w > W + 2 || t.y + t.h > H + 2) {
      out.push(`越界:「${t.s.slice(0, 24)}」 x ${t.x.toFixed(0)}–${(t.x + t.w).toFixed(0)} / y ${t.y.toFixed(0)}–${(t.y + t.h).toFixed(0)},画布 ${W}×${H}`);
    }
  }
  for (let i = 0; i < texts.length; i += 1) {
    for (let j = i + 1; j < texts.length; j += 1) {
      if (hit(texts[i], texts[j], 2)) out.push(`互压:「${texts[i].s.slice(0, 20)}」与「${texts[j].s.slice(0, 20)}」`);
    }
  }
  // 只有画在文字之后的填充图形才盖得住它;画在之前的是它的底板
  for (const t of texts) {
    for (const s of shapes) {
      if (s.at > t.at && hit(t, s, 2)) {
        out.push(`被盖住:「${t.s.slice(0, 24)}」压在一个后画的 ${s.kind}(${s.fill})下面`);
        break;
      }
    }
  }
  return out;
}

const invokedDirectly = process.argv[1] && path.resolve(process.argv[1]).endsWith('svg-check.mjs');
if (!invokedDirectly) { /* 被 import,只导出 checkFile */ } else {
const targets = process.argv.slice(2).flatMap((p) => {
  const st = fs.statSync(p);
  return st.isDirectory()
    ? fs.readdirSync(p).filter((f) => f.endsWith('.svg')).sort().map((f) => path.join(p, f))
    : [p];
});
if (!targets.length) {
  console.error('用法: node scripts/svg-check.mjs <文件或目录…>');
  process.exit(2);
}
let bad = 0;
for (const f of targets) {
  const problems = checkFile(f);
  if (problems.length) {
    bad += 1;
    console.log(`✗ ${path.basename(f)}`);
    for (const p of problems) console.log(`    ${p}`);
  } else {
    console.log(`✓ ${path.basename(f)}`);
  }
}
console.log(bad ? `\n${bad}/${targets.length} 张有问题` : `\n${targets.length} 张全部通过`);
process.exit(bad ? 1 : 0);
}
