#!/usr/bin/env node
// 把 archify 交付的交互式 HTML 导出为独立 SVG,复用页面自带的「导出 → SVG」逻辑。
// 用法: node scripts/archify-export.mjs <delivered.html> <out.svg>
// 依赖: ~/.claude/skills/archify(可用 ARCHIFY_HOME 覆盖)里的无头 Chrome CDP 封装,与 visual-check 同源。
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [,, htmlPath, outPath] = process.argv;
if (!htmlPath || !outPath) {
  console.error('用法: node scripts/archify-export.mjs <delivered.html> <out.svg>');
  process.exit(2);
}
const archifyHome = process.env.ARCHIFY_HOME || path.join(os.homedir(), '.claude', 'skills', 'archify');
const { ChromeVisualBrowser, findChrome } = await import(pathToFileURL(path.join(archifyHome, 'bin', 'visual-check.mjs')).href);
const chromePath = findChrome();
if (!chromePath) { console.error('找不到 Chrome,可设置 ARCHIFY_CHROME'); process.exit(3); }

const browser = new ChromeVisualBrowser(chromePath);
let exitCode = 0;
try {
  const sessionId = await browser.sessionPromise;
  const evaluate = async (expression, awaitPromise = false) => {
    const r = await browser.cdp.send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true }, sessionId, 60000);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text || 'Runtime.evaluate failed');
    return r.result?.value;
  };
  await browser.cdp.send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false }, sessionId);
  const url = new URL(pathToFileURL(path.resolve(htmlPath)).href);
  url.searchParams.set('theme', 'light');
  const loaded = browser.cdp.waitFor('Page.loadEventFired', sessionId);
  const nav = await browser.cdp.send('Page.navigate', { url: url.href }, sessionId);
  if (nav.errorText) throw new Error('导航失败: ' + nav.errorText);
  await loaded;
  // 与 visual-check 相同的稳定等待:静止动画、字体就绪、阅读布局稳定
  await evaluate(`(function () {
    document.documentElement.setAttribute('data-motion', 'still');
    var fontsReady = document.fonts && document.fonts.ready ? document.fonts.ready.catch(function () {}) : Promise.resolve();
    return fontsReady.then(function () {
      if (window.Archify && Archify.readerLayout && typeof Archify.readerLayout.whenStable === 'function') return Archify.readerLayout.whenStable();
    }).then(function () {
      return new Promise(function (resolve) { requestAnimationFrame(function () { requestAnimationFrame(resolve); }); });
    });
  })()`, true);
  // 劫持 Blob 下载,点击导出菜单的 SVG 项,取回字符串
  const svg = await evaluate(`(function () {
    return new Promise(function (resolve, reject) {
      var captured = null;
      URL.createObjectURL = function (blob) { captured = blob; return 'blob:captured'; };
      URL.revokeObjectURL = function () {};
      HTMLAnchorElement.prototype.click = function () {};
      var btn = document.querySelector('button[data-format="svg"]');
      if (!btn) return reject(new Error('no svg export button'));
      btn.click();
      var tries = 0;
      (function poll() {
        if (captured) return captured.text().then(resolve, reject);
        if (++tries > 400) return reject(new Error('timeout waiting for export blob'));
        setTimeout(poll, 25);
      })();
    });
  })()`, true);
  if (typeof svg !== 'string' || !(svg.startsWith('<svg') || svg.startsWith('<?xml'))) throw new Error('导出内容不是 SVG: ' + String(svg).slice(0, 80));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, svg);
  console.log(JSON.stringify({ ok: true, out: outPath, bytes: Buffer.byteLength(svg), hasStyle: svg.includes('<style'), autoTheme: svg.includes('prefers-color-scheme') }));
} catch (error) {
  console.error('导出失败:', error.message);
  exitCode = 1;
} finally {
  await browser.close();
}
process.exit(exitCode);
