import { NextResponse } from 'next/server'
import { isValidRef, saveNote, setHighFreq, setMastered } from '@/lib/questions'

export async function PATCH(req: Request) {
  let body: unknown
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }
  const { category, file, mastered, note, highfreq } = (body ?? {}) as Record<string, unknown>

  if (typeof category !== 'string' || typeof file !== 'string')
    return NextResponse.json({ error: 'category/file 必填' }, { status: 400 })
  if (mastered === undefined && note === undefined && highfreq === undefined)
    return NextResponse.json({ error: 'mastered/note/highfreq 至少提供一项' }, { status: 400 })
  if (mastered !== undefined && typeof mastered !== 'boolean')
    return NextResponse.json({ error: 'mastered 必须是布尔值' }, { status: 400 })
  if (highfreq !== undefined && typeof highfreq !== 'boolean')
    return NextResponse.json({ error: 'highfreq 必须是布尔值' }, { status: 400 })
  if (note !== undefined && typeof note !== 'string')
    return NextResponse.json({ error: 'note 必须是字符串' }, { status: 400 })
  if (!isValidRef(category, file))
    return NextResponse.json({ error: '题目不存在' }, { status: 400 })

  try {
    if (mastered !== undefined) setMastered(category, file, mastered)
    if (highfreq !== undefined) setHighFreq(category, file, highfreq)
    if (note !== undefined) saveNote(category, file, note)
  } catch (e) {
    return NextResponse.json({ error: `写入失败:${(e as Error).message}` }, { status: 500 })
  }
  return NextResponse.json({ ok: true })
}
