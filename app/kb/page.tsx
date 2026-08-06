import Link from 'next/link'
import { listArticleTopics, listKbCategories } from '@/lib/knowledge'

export const dynamic = 'force-dynamic'

export default function KbIndexPage() {
  const cats = listKbCategories().map((c) => ({ name: c, topics: listArticleTopics(c) }))
  const total = cats.reduce((s, c) => s + c.topics.length, 0)
  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">知识库</h1>
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
      </div>
      <p className="mt-2 text-sm text-gray-500">
        每篇对应一个 topic 主题的整体讲解;题目是切片,文章是全景。共 {total} 篇。
      </p>
      {total === 0 ? (
        <p className="mt-10 text-gray-500">
          还没有文章。在 knowledge/&lt;分类&gt;/&lt;topic&gt;.md 新建即可,规范见 docs/question-authoring.md。
        </p>
      ) : (
        cats
          .filter((c) => c.topics.length > 0)
          .map((c) => (
            <section key={c.name} className="mt-10">
              <h2 className="mb-3 font-mono text-xs tracking-widest text-gray-400">{c.name}</h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {c.topics.map((t) => (
                  <Link
                    key={t}
                    href={`/kb/${encodeURIComponent(c.name)}/${encodeURIComponent(t)}`}
                    className="block border border-gray-200 bg-white px-4 py-3 font-semibold transition-colors hover:border-gray-400"
                  >
                    {t}
                  </Link>
                ))}
              </div>
            </section>
          ))
      )}
    </main>
  )
}
