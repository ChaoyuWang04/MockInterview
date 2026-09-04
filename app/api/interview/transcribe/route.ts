import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const VOICE_URL = process.env.INTERVIEW_VOICE_URL ?? 'http://127.0.0.1:8700'

/**
 * 转写。透传给语音服务。
 *
 * `hotwords` 是这套系统相对通用听写工具的独有优势:每道题会出现哪些技术术语,
 * 我们是知道的(文章名 + 题干里的术语),喂进去能提高召回。
 * 只传规范写法 —— 变体反而会把本地 Whisper 带跑偏,见 useVoice.ts 的实测数字。
 */
export async function POST(req: Request) {
  let form: FormData
  try {
    form = await req.formData()
  } catch {
    return NextResponse.json({ error: '请求不是 multipart 表单' }, { status: 400 })
  }
  const file = form.get('file')
  if (!(file instanceof Blob)) return NextResponse.json({ error: '缺少音频文件' }, { status: 400 })

  const out = new FormData()
  out.append('file', file, 'audio.wav')
  out.append('model', String(form.get('model') ?? 'qwen3-asr-flash'))
  const hot = form.get('hotwords')
  if (hot) out.append('hotwords', String(hot))

  try {
    const res = await fetch(`${VOICE_URL}/v1/audio/transcriptions`, {
      method: 'POST',
      body: out,
      // 和 voice-server 里那次云端调用的 300s 对齐 —— 代理先超时的话,
      // 你看到的是「语音服务不可达」,真正的原因(体积超限之类)就永远看不到了
      signal: AbortSignal.timeout(310_000),
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.ok ? 200 : 502 })
  } catch (e) {
    return NextResponse.json({ error: `语音服务不可达:${(e as Error).message}` }, { status: 502 })
  }
}
