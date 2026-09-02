import { NextResponse } from 'next/server'
import { readCached, cacheStats } from '@/lib/interview/ttsCache'

export const dynamic = 'force-dynamic'

const VOICE_URL = process.env.INTERVIEW_VOICE_URL ?? 'http://127.0.0.1:8700'

/** 列出语音服务里可用的 TTS 模型/音色与 STT 后端。前端的下拉框读这个。 */
export async function GET() {
  try {
    const res = await fetch(`${VOICE_URL}/voices`, { signal: AbortSignal.timeout(5000) })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return NextResponse.json({ up: true, cache: cacheStats(), ...(await res.json()) })
  } catch (e) {
    // 语音服务没起也不该让页面挂掉 —— 打字面试照常可用
    return NextResponse.json({
      up: false,
      error: (e as Error).message,
      hint: '语音服务没在跑。启动:.venv-voice/bin/python scripts/voice-server.py',
      tts: [],
      stt: [],
    })
  }
}

/** 合成语音。透传给语音服务,把 wav 原样返回给浏览器。 */
export async function POST(req: Request) {
  let body: Record<string, unknown>
  try {
    body = (await req.json()) as Record<string, unknown>
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }
  const text = String(body.input ?? '').trim()
  if (!text) return NextResponse.json({ error: 'input 是空的' }, { status: 400 })

  // 先查预渲染缓存 —— 主问全部命中,延迟 0;追问是新文本,必然未命中走合成
  const cfg = {
    model: body.model as string | undefined,
    voice: body.voice as string | undefined,
    instruct: body.instruct as string | undefined,
    speed: body.speed as number | undefined,
    normalize: body.normalize as boolean | undefined,
  }
  const hit = readCached(text, { ...cfg, normalize: cfg.normalize ?? true })
  if (hit) {
    return new Response(new Uint8Array(hit.buf), {
      headers: {
        'Content-Type': hit.type,
        'X-Elapsed-Ms': '0',
        'X-Cache': 'hit',
      },
    })
  }

  try {
    const res = await fetch(`${VOICE_URL}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // ⚠️ 这里要把前端给的**每一个**参数都带上。
      // 踩过:漏了 instruct 和 normalize,结果音色描述改了不生效(服务端回落到默认音色)、
      // 读音修正开关也是死的 —— 而且两处都不报错,只是行为对不上。
      body: JSON.stringify({
        input: text,
        model: body.model,
        voice: body.voice,
        instruct: body.instruct,
        speed: body.speed,
        temperature: body.temperature,
        normalize: body.normalize,
      }),
      signal: AbortSignal.timeout(120_000),
    })
    if (!res.ok) {
      return NextResponse.json({ error: `语音服务 ${res.status}: ${(await res.text()).slice(0, 200)}` }, { status: 502 })
    }
    // 语音服务发什么 X- 头就原样透传,别在这里挑 ——
    // 之前手写白名单漏了 X-Spoken,前端「实际念的是…」那一行就永远不显示。
    // 同一类疏忽这里已经犯过两次(请求体漏 instruct/normalize、响应头漏 X-Spoken)。
    const headers = new Headers({ 'Content-Type': 'audio/wav' })
    res.headers.forEach((v, k) => {
      if (k.toLowerCase().startsWith('x-')) headers.set(k, v)
    })
    headers.set('X-Cache', 'miss')
    return new Response(await res.arrayBuffer(), { headers })
  } catch (e) {
    return NextResponse.json({ error: `语音服务不可达:${(e as Error).message}` }, { status: 502 })
  }
}
