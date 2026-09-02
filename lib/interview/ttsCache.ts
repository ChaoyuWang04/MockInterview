import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'

/**
 * 预渲染音频的缓存。
 *
 * 主问的文本是已知且固定的,所以能提前渲染好;命中缓存时播放延迟是 0,
 * 而现场合成要 2–6 秒。追问是现场生成的,躲不掉,所以只优化主问这一半。
 *
 * key 里必须包含**所有影响音频的参数** —— 换音色/语速/读音修正就是另一份缓存,
 * 不会串味,也不用手动清理。
 */
export interface TtsCacheKey {
  model?: string
  voice?: string
  instruct?: string
  speed?: number
  normalize?: boolean
}

export function cacheDir(): string {
  return path.join(process.cwd(), 'interview', 'cache', 'tts')
}

/**
 * 规范化后再哈希。
 *
 * 为什么必须规范化:`voice: ''` 和 `voice: undefined` 会产生不同的 key,
 * 于是链路上任何一跳漏传一个字段,缓存就**静默全部失效** ——
 * 不报错、不变慢,只是每次都重新合成,你根本发现不了。
 * 所以缺省值在这里统一补齐,而不是指望调用方都传全。
 */
export function cacheKey(text: string, cfg: TtsCacheKey): string {
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

/** 命中就返回 {buf, contentType},否则 null。mp3 优先(体积只有 wav 的 1/6) */
export function readCached(text: string, cfg: TtsCacheKey): { buf: Buffer; type: string } | null {
  const key = cacheKey(text, cfg)
  for (const [ext, type] of [
    ['mp3', 'audio/mpeg'],
    ['wav', 'audio/wav'],
  ] as const) {
    const p = path.join(cacheDir(), `${key}.${ext}`)
    if (fs.existsSync(p)) return { buf: fs.readFileSync(p), type }
  }
  return null
}

export function cacheStats(): { count: number; bytes: number } {
  const dir = cacheDir()
  if (!fs.existsSync(dir)) return { count: 0, bytes: 0 }
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.mp3') || f.endsWith('.wav'))
  return {
    count: files.length,
    bytes: files.reduce((s, f) => s + fs.statSync(path.join(dir, f)).size, 0),
  }
}
