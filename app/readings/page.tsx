import Link from 'next/link'
import { listReadingTopics } from '@/lib/readings'

export const dynamic = 'force-dynamic'

export default function ReadingsIndexPage() {
  const topics = listReadingTopics()
  const total = topics.reduce((sum, topic) => sum + topic.readings.length, 0)

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
        topics.map((topic) => (
          <section key={topic.title} className="mt-12">
            <h2 className="mb-3 flex items-baseline gap-2 border-b border-gray-200 pb-1 text-lg font-bold">
              {topic.title}
              <span className="font-mono text-xs font-normal text-gray-400">
                {topic.readings.length} 篇
              </span>
            </h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {topic.readings.map((reading) => (
                <Link
                  key={`${topic.title}/${reading.slug}`}
                  href={`/readings/${encodeURIComponent(topic.title)}/${encodeURIComponent(reading.slug)}`}
                  className="block border border-gray-200 bg-white px-5 py-4 transition-colors hover:border-gray-400"
                >
                  <span className="flex items-start justify-between gap-3">
                    <span className="font-semibold">{reading.title}</span>
                    <span
                      title="机构，来自 readings/index.md 的第二列，不是目录名"
                      className="mt-0.5 shrink-0 border border-gray-300 bg-gray-50 px-1 font-mono text-[10px] font-normal text-gray-500"
                    >
                      {reading.org}
                    </span>
                  </span>
                  <span className="mt-2 flex items-center justify-between gap-4 font-mono text-xs text-gray-400">
                    <span>{reading.releaseDate} 首发</span>
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
