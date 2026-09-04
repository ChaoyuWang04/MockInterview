import Link from 'next/link'
import { notFound } from 'next/navigation'
import Markdown from '@/components/Markdown'
import { readSession } from '@/lib/interview/session'

export const dynamic = 'force-dynamic'

/**
 * 把整篇拆成复盘和逐轮记录。
 * 拆不开(旧文件、没生成复盘)时整篇当复盘渲染 —— 宁可多显示,也不要空白页。
 */
function split(body: string): { review: string; turns: string } {
  const i = body.indexOf('## 逐轮记录')
  if (i < 0) return { review: body.replace(/^##\s*复盘\s*/m, '').trim(), turns: '' }
  return {
    review: body.slice(0, i).replace(/^##\s*复盘\s*/m, '').trim(),
    turns: body.slice(i + '## 逐轮记录'.length).trim(),
  }
}

export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const found = readSession(decodeURIComponent(id))
  if (!found) notFound()
  const { summary: s, body } = found
  // 丢掉第一个 `## ` 之前的头部(`# 标题` + `- 开始:` 那几行)—— 页头已经显示过了。
  // ⚠️ 不能用 `^- .*$` 全局删元信息行:逐轮记录里的 `- ✅/❌ 要点` 也是这个形状,会被一起删掉。
  const firstSection = body.search(/^## /m)
  const { review, turns } = split(firstSection >= 0 ? body.slice(firstSection) : body)

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-6 flex items-baseline justify-between gap-4">
        <h1 className="text-2xl font-bold">{s.主题}</h1>
        <Link href="/interview/sessions" className="shrink-0 text-xs text-gray-400 hover:text-gray-700">
          ← 复盘列表
        </Link>
      </div>

      <div className="border-b border-gray-200 pb-4 font-mono text-[11px] text-gray-500">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span>
            {s.date} {s.time}
          </span>
          <span className="text-gray-300">·</span>
          <span>{s.简历}</span>
          <span className="text-gray-300">·</span>
          <span>{s.模式}</span>
          <span className="text-gray-300">·</span>
          <span>{s.轮次} 轮</span>
          <span className="text-gray-300">·</span>
          <span>
            要点命中 {s.hit}/{s.points}
          </span>
        </div>
        {s.覆盖.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {s.覆盖.map((a) => (
              <span key={a} className="bg-gray-100 px-2 py-0.5 text-gray-600">
                {a}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="prose prose-sm mt-8 max-w-none">
        <Markdown>{review || '(这场没有生成复盘)'}</Markdown>
      </div>

      {turns && (
        <details className="mt-10 border-t border-gray-200 pt-4">
          <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-900">
            逐轮记录 · {s.轮次} 轮(语音转写原文,可能有听错的地方)
          </summary>
          <div className="prose prose-sm mt-4 max-w-none">
            <Markdown>{turns}</Markdown>
          </div>
        </details>
      )}
    </main>
  )
}
