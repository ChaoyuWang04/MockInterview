import Link from 'next/link'
import { notFound } from 'next/navigation'
import Markdown from '@/components/Markdown'
import { getArticleBySegments, stripOrder } from '@/lib/knowledge'
import { listCategories, loadCategory } from '@/lib/questions'

export const dynamic = 'force-dynamic'

export default async function ArticlePage({ params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params
  const segments = slug.map(decodeURIComponent)
  const md = getArticleBySegments(segments)
  if (md === null) notFound()

  const title = stripOrder(segments[segments.length - 1].replace(/\.md$/, ''))
  const breadcrumb = segments.slice(0, -1).map(stripOrder)

  // 关联题目:全库扫描 topic 第一段等于本文标题的题
  const related = listCategories().flatMap((c) =>
    loadCategory(c)
      .map((q, i) => ({ q, n: i + 1, category: c }))
      .filter(({ q }) => !q.error && q.meta.topic?.split('/')[0]?.trim() === title),
  )

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
          {breadcrumb.join(' · ')}
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
            本主题的题目({related.length})—— 在对应分类的列表页点开
          </h2>
          <ul className="mt-3 space-y-1.5 text-sm">
            {related.map(({ q, n, category }) => (
              <li key={`${category}/${q.file}`} className="flex items-baseline gap-2">
                <Link
                  href={`/${encodeURIComponent(category)}`}
                  className="font-mono text-xs text-blue-600 hover:underline"
                >
                  {category} #{n}
                </Link>
                <span className={q.meta.mastered ? 'text-green-600' : 'text-gray-700'}>
                  {q.meta.mastered ? '✓ ' : ''}
                  {q.meta.summary ?? q.file}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  )
}
