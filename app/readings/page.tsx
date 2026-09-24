import type { Metadata } from 'next'
import Link from 'next/link'
import LibraryIndex from '@/components/LibraryIndex'
import { readingViews } from '@/lib/library-views'

export const dynamic = 'force-dynamic'

export const metadata: Metadata = { title: '日常研读' }

export default function ReadingsIndexPage() {
  const views = readingViews()
  const total = views.total

  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">日常研读</h1>
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
      </div>
      <p className="mt-2 text-sm leading-6 text-gray-500">
        关注但不必精读的论文、网页与视频字幕，用来扫方向、筛未来可能的方向。方向就是目录；每个方向按首发日从新到旧，条幅标出机构。写作契约与
        {' '}
        <Link href="/reports" className="underline hover:text-gray-900">
          报告解读
        </Link>
        {' '}
        完全一致。当前共 {total} 篇。
      </p>

      {total === 0 ? (
        <div className="mt-10 border border-dashed border-gray-300 bg-white p-6 text-sm leading-7 text-gray-500">
          还没有已发布的解读。原件放到 readings/_src/&lt;方向&gt;/，完成的长文放到
          readings/&lt;方向&gt;/，并在 readings/index.md 对应方向登记一行；完整流程见
          docs/10-材料解读流程.md。
        </div>
      ) : (
        <LibraryIndex
          library="readings"
          badgeTitle="机构，来自 readings/index.md 的第二列，不是目录名"
          byTopic={views.byTopic}
          byTime={views.byTime}
          initialRead={views.initialRead}
        />
      )}
    </main>
  )
}
