import Link from 'next/link'
import { listSessionSummaries, type SessionSummary } from '@/lib/interview/session'

export const dynamic = 'force-dynamic'

/** 命中率条。和首页分类卡片同一套视觉,不另起风格。 */
function HitBar({ hit, points }: { hit: number; points: number }) {
  const pct = points ? Math.round((hit / points) * 100) : 0
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 bg-gray-100">
        <div className="h-full bg-blue-600" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[11px] text-gray-400">
        {hit}/{points}
      </span>
    </div>
  )
}

function Row({ s }: { s: SessionSummary }) {
  return (
    <Link
      href={`/interview/sessions/${s.id}`}
      className="block border border-gray-200 bg-white p-4 transition-colors hover:border-gray-400"
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="truncate font-semibold">{s.主题}</span>
        <span className="shrink-0 font-mono text-xs text-gray-400">{s.time}</span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-gray-500">
        <span>{s.简历}</span>
        <span className="text-gray-300">·</span>
        <span>{s.模式}</span>
        <span className="text-gray-300">·</span>
        <span>{s.轮次} 轮</span>
        <HitBar hit={s.hit} points={s.points} />
      </div>
    </Link>
  )
}

export default function SessionsPage() {
  const sessions = listSessionSummaries()

  // 按日期分组。listSessionSummaries 已按新→旧排好,这里保持插入序即可
  const byDate = new Map<string, SessionSummary[]>()
  for (const s of sessions) {
    const bucket = byDate.get(s.date)
    if (bucket) bucket.push(s)
    else byDate.set(s.date, [s])
  }

  const totalHit = sessions.reduce((n, s) => n + s.hit, 0)
  const totalPoints = sessions.reduce((n, s) => n + s.points, 0)

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">面试复盘</h1>
        <Link href="/interview" className="text-xs text-gray-400 hover:text-gray-700">
          ← 模拟面试
        </Link>
      </div>

      {sessions.length === 0 ? (
        <p className="mt-10 text-sm text-gray-500">
          还没有复盘。去 <Link href="/interview" className="underline">模拟面试</Link> 面一场,
          结束时会自动生成复盘并存进 <code className="bg-gray-100 px-1">interview/sessions/</code>。
        </p>
      ) : (
        <>
          <p className="mt-2 font-mono text-xs text-gray-400">
            {sessions.length} 场 · 累计 {sessions.reduce((n, s) => n + s.轮次, 0)} 轮 · 要点命中{' '}
            {totalHit}/{totalPoints}
          </p>

          {[...byDate].map(([date, items]) => (
            <section key={date} className="mt-8">
              <div className="mb-3 flex items-baseline justify-between border-b border-gray-200 pb-1">
                <h2 className="font-mono text-sm text-gray-600">{date}</h2>
                <span className="font-mono text-xs text-gray-400">{items.length} 场</span>
              </div>
              <div className="space-y-2">
                {items.map((s) => (
                  <Row key={s.id} s={s} />
                ))}
              </div>
            </section>
          ))}
        </>
      )}
    </main>
  )
}
