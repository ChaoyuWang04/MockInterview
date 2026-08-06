import Link from 'next/link'
import { notFound } from 'next/navigation'
import Markdown from '@/components/Markdown'
import { getArticle } from '@/lib/knowledge'
import { listCategories, loadCategory } from '@/lib/questions'

export const dynamic = 'force-dynamic'

export default async function ArticlePage({
  params,
}: {
  params: Promise<{ category: string; topic: string }>
}) {
  const { category: rawC, topic: rawT } = await params
  const category = decodeURIComponent(rawC)
  const topic = decodeURIComponent(rawT)
  const md = getArticle(category, topic)
  if (md === null) notFound()

  const related = listCategories().includes(category)
    ? loadCategory(category)
        .map((q, i) => ({ q, n: i + 1 }))
        .filter(({ q }) => !q.error && q.meta.topic?.split('/')[0]?.trim() === topic)
    : []

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <div className="mb-8 flex items-center gap-4 text-sm text-gray-500">
        <Link href="/kb" className="hover:text-gray-900">
          ← 返回知识库
        </Link>
        <span className="text-gray-300">|</span>
        <Link href="/" className="hover:text-gray-900">
          返回主页
        </Link>
        <span className="ml-auto font-mono text-xs tracking-widest text-gray-400">
          {category} · {topic}
        </span>
      </div>

      <article className="border border-gray-200 bg-white p-8">
        <div className="prose max-w-none">
          <Markdown>{md}</Markdown>
        </div>
      </article>

      {related.length > 0 && (
        <section className="mt-6 border border-gray-200 bg-white p-6">
          <h2 className="font-mono text-xs tracking-widest text-gray-400">
            本主题的题目({related.length})—— 在
            <Link href={`/${encodeURIComponent(category)}`} className="mx-1 text-blue-600 hover:underline">
              {category} 刷题页
            </Link>
            右侧导航栏点击对应题号
          </h2>
          <ul className="mt-3 space-y-1.5 text-sm">
            {related.map(({ q, n }) => (
              <li key={q.file} className="flex items-baseline gap-2">
                <span className={`font-mono text-xs ${q.meta.mastered ? 'text-green-600' : 'text-gray-400'}`}>
                  #{n}{q.meta.mastered ? ' ✓' : ''}
                </span>
                <span className="text-gray-700">{q.meta.summary ?? q.file}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  )
}
