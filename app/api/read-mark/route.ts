import { NextResponse } from 'next/server'
import { setReadMark } from '@/lib/read-marks'
import { isValidReading, readingsRoot } from '@/lib/readings'
import { isValidReport, reportsRoot } from '@/lib/reports'

/** 两个解读库卡片上的「已读」勾选:只增删 `<库根>/_已读.md` 里对应的那一行 */
export async function PATCH(req: Request) {
  let body: unknown
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }
  const { library, key, read } = (body ?? {}) as Record<string, unknown>

  if ((library !== 'reports' && library !== 'readings') || typeof key !== 'string' || typeof read !== 'boolean')
    return NextResponse.json(
      { error: 'library 必须是 reports 或 readings,key 必填(字符串),read 必填(布尔)' },
      { status: 400 },
    )
  const parts = key.split('/')
  const exists =
    parts.length === 2 &&
    (library === 'reports' ? isValidReport(parts[0], parts[1]) : isValidReading(parts[0], parts[1]))
  if (!exists) return NextResponse.json({ error: `${library} 里没有已发布的 ${key}` }, { status: 400 })

  try {
    setReadMark(library === 'reports' ? reportsRoot() : readingsRoot(), key, read)
  } catch (e) {
    return NextResponse.json({ error: `写入失败:${(e as Error).message}` }, { status: 500 })
  }
  return NextResponse.json({ ok: true })
}
