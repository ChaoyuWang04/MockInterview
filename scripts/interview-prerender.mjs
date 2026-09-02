// 把 94 道主问的题干预渲染成语音:npm run interview:tts
//
// 为什么值得做:主问的文本是**已知且固定**的,现场生成要等 2–6 秒,
// 预渲染后读缓存就是 0 延迟。而追问是现场生成的,躲不掉,所以只优化主问这一半。
//
// 体积很小:99 道题干合计约 8.8 分钟音频,mp3 64k 约 4 MB。
// 换音色/语速就重跑一次(RTF 0.6,全量约 5 分钟),缓存按参数哈希分目录,互不覆盖。
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import matter from 'gray-matter'
import { execFileSync } from 'node:child_process'

const VOICE_URL = process.env.INTERVIEW_VOICE_URL ?? 'http://127.0.0.1:8700'
const CACHE = path.join(process.cwd(), 'interview', 'cache', 'tts')

const visible = (n) => !n.startsWith('.') && !n.startsWith('_')

function walk(dir, acc = []) {
  if (!fs.existsSync(dir)) return acc
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!visible(e.name)) continue
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p, acc)
    else if (e.name.endsWith('.md')) acc.push(p)
  }
  return acc
}

/** 题干 = `## 题目` 分区。念的是原文,不改写 —— 保留真题口气,缓存也才稳定 */
function questionText(file) {
  const body = matter(fs.readFileSync(file, 'utf8')).content
  const m = body.match(/^## 题目\n([\s\S]*?)(?=^## |\Z)/m)
  return m ? m[1].trim() : ''
}

/** 缓存 key:文本 + 全部影响音频的参数。
 *  ⚠️ 必须和 lib/interview/ttsCache.ts 的实现逐字一致 ——
 *  两边由 tests/interview.test.ts 的用例钉住,改一边必须改另一边。 */
export function cacheKey(text, cfg) {
  const sig = JSON.stringify([
    text.trim(),
    cfg.model ?? '',
    cfg.voice ?? '',
    cfg.instruct ?? '',
    Number(cfg.speed ?? 1),
    cfg.normalize !== false,
  ])
  return crypto.createHash('sha256').update(sig).digest('hex').slice(0, 16)
}

async function synth(text, cfg) {
  const res = await fetch(`${VOICE_URL}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: text, ...cfg }),
  })
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 160)}`)
  return {
    buf: Buffer.from(await res.arrayBuffer()),
    seconds: Number(res.headers.get('x-audio-seconds') ?? 0),
    ms: Number(res.headers.get('x-elapsed-ms') ?? 0),
  }
}

async function main() {
  const defaults = await fetch(`${VOICE_URL}/voices`)
    .then((r) => r.json())
    .catch(() => null)
  if (!defaults?.defaults) {
    console.error(`语音服务不可达:${VOICE_URL}\n先跑 bash scripts/voice-start.sh`)
    process.exit(1)
  }
  const cfg = {
    model: defaults.defaults.tts,
    voice: defaults.defaults.voice ?? '',
    instruct: defaults.defaults.instruct,
    speed: defaults.defaults.speed,
    temperature: defaults.defaults.temperature,
    normalize: true,
  }

  const files = walk(path.join(process.cwd(), 'questions')).filter(
    (f) => !f.includes(`${path.sep}手撕代码${path.sep}`), // 口头面试念不了代码题
  )
  fs.mkdirSync(CACHE, { recursive: true })

  console.log(`\n预渲染主问 · ${files.length} 道`)
  console.log(`音色:${cfg.instruct.slice(0, 40)}…`)
  console.log(`语速 ${cfg.speed}× · 稳定度 ${cfg.temperature}\n`)

  let done = 0, skipped = 0, failed = 0, secs = 0, bytes = 0
  const t0 = Date.now()
  for (const file of files) {
    const text = questionText(file)
    const rel = path.relative(path.join(process.cwd(), 'questions'), file)
    if (!text) {
      console.log(`  ⏭  ${rel}(没有 ## 题目)`)
      continue
    }
    const key = cacheKey(text, cfg)
    const out = path.join(CACHE, `${key}.mp3`)
    if (fs.existsSync(out) || fs.existsSync(path.join(CACHE, `${key}.wav`))) {
      skipped++
      continue
    }
    try {
      const { buf, seconds, ms } = await synth(text, cfg)
      // 转 mp3 64k:体积只有 wav 的 1/6(43MB → 7.4MB),念题这个用途完全够
      const mp3 = execFileSync(
        'ffmpeg',
        ['-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
         '-codec:a', 'libmp3lame', '-b:a', '64k', '-ac', '1', '-f', 'mp3', 'pipe:1'],
        { input: buf, maxBuffer: 1 << 28 },
      )
      fs.writeFileSync(out, mp3)
      done++; secs += seconds; bytes += mp3.length
      process.stdout.write(
        `\r  ${String(done + skipped).padStart(3)}/${files.length}  ${ms}ms  ${rel.slice(0, 44).padEnd(44)}`,
      )
    } catch (e) {
      failed++
      console.log(`\n  ❌ ${rel}: ${e.message}`)
    }
  }
  const mins = (Date.now() - t0) / 60000
  console.log(`\n\n  新渲染 ${done} · 已存在 ${skipped} · 失败 ${failed}`)
  console.log(`  音频 ${(secs / 60).toFixed(1)} 分钟 · ${(bytes / 1048576).toFixed(1)} MB · 耗时 ${mins.toFixed(1)} 分钟`)
  console.log(`  缓存在 ${path.relative(process.cwd(), CACHE)}/\n`)
  if (failed) process.exitCode = 1
}

if (import.meta.url === `file://${process.argv[1]}`) main()
