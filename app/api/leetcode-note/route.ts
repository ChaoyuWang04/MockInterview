import { NextResponse } from 'next/server'
import { isValidSlug, saveNote } from '@/lib/leetcode'

export async function PATCH(req: Request) {
  let body: unknown
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }
  const { slug, note } = (body ?? {}) as Record<string, unknown>

  if (typeof slug !== 'string' || typeof note !== 'string')
    return NextResponse.json({ error: 'slug/note 必填且为字符串' }, { status: 400 })
  if (!isValidSlug(slug))
    return NextResponse.json({ error: '题目不存在于 hot100 清单' }, { status: 400 })

  try {
    saveNote(slug, note)
  } catch (e) {
    return NextResponse.json({ error: `写入失败:${(e as Error).message}` }, { status: 500 })
  }
  return NextResponse.json({ ok: true })
}
