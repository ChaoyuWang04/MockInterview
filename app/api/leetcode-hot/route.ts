import { NextResponse } from 'next/server'
import { isValidSlug, setHighFreq } from '@/lib/leetcode'

export async function PATCH(req: Request) {
  let body: unknown
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }
  const { slug, hot } = (body ?? {}) as Record<string, unknown>

  if (typeof slug !== 'string' || typeof hot !== 'boolean')
    return NextResponse.json({ error: 'slug 必填(字符串)、hot 必填(布尔)' }, { status: 400 })
  if (!isValidSlug(slug))
    return NextResponse.json({ error: '题目不存在于 hot100 清单' }, { status: 400 })

  try {
    setHighFreq(slug, hot)
  } catch (e) {
    return NextResponse.json({ error: `写入失败:${(e as Error).message}` }, { status: 500 })
  }
  return NextResponse.json({ ok: true })
}
