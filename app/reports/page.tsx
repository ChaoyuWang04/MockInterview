import Link from 'next/link'
import { listReportTopics } from '@/lib/reports'

export const dynamic = 'force-dynamic'

export default function ReportsIndexPage() {
  const topics = listReportTopics()
  const total = topics.reduce((sum, topic) => sum + topic.reports.length, 0)

  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">报告解读</h1>
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
      </div>
      <p className="mt-2 text-sm leading-6 text-gray-500">
        按技术方向整理公开的基模 Technical Report
        与核心技术论文。每个方向按首发日从新到旧;条幅标出主要归属方。每篇用一篇完整长文讲清设计、原因和可迁移的部分。当前共
        {' '}
        {total} 篇。
      </p>

      {total === 0 ? (
        <div className="mt-10 border border-dashed border-gray-300 bg-white p-6 text-sm leading-7 text-gray-500">
          还没有已发布的解读。PDF 原件放到 papers/&lt;公司&gt;/，完成的长文放到
          reports/&lt;公司&gt;/，并在 reports/index.md 登记方向；完整流程见 docs/10-材料解读流程.md。
        </div>
      ) : (
        topics.map((topic) => (
          <section key={topic.title} className="mt-12">
            <h2 className="mb-3 flex items-baseline gap-2 border-b border-gray-200 pb-1 text-lg font-bold">
              {topic.title}
              <span className="font-mono text-xs font-normal text-gray-400">
                {topic.reports.length} 篇
              </span>
            </h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {topic.reports.map((report) => (
                <Link
                  key={`${report.company}/${report.slug}`}
                  href={`/reports/${encodeURIComponent(report.company)}/${encodeURIComponent(report.slug)}`}
                  className="block border border-gray-200 bg-white px-5 py-4 transition-colors hover:border-gray-400"
                >
                  <span className="flex items-start justify-between gap-3">
                    <span className="font-semibold">{report.title}</span>
                    <span
                      title="主要归属方,即 reports/ 下的目录名"
                      className="mt-0.5 shrink-0 border border-gray-300 bg-gray-50 px-1 font-mono text-[10px] font-normal text-gray-500"
                    >
                      {report.company}
                    </span>
                  </span>
                  <span className="mt-2 flex items-center justify-between gap-4 font-mono text-xs text-gray-400">
                    <span>{report.releaseDate} 首发</span>
                    <span>阅读全文 →</span>
                  </span>
                </Link>
              ))}
            </div>
          </section>
        ))
      )}
    </main>
  )
}
