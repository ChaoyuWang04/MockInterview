import Link from 'next/link'
import LibraryIndex from '@/components/LibraryIndex'
import { reportViews } from '@/lib/library-views'

export const dynamic = 'force-dynamic'

export default function ReportsIndexPage() {
  const views = reportViews()
  const total = views.total

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
        <LibraryIndex
          library="reports"
          badgeTitle="主要归属方,即 reports/ 下的目录名"
          byTopic={views.byTopic}
          byTime={views.byTime}
          initialRead={views.initialRead}
        />
      )}
    </main>
  )
}
