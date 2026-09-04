import Link from 'next/link'
import { notFound } from 'next/navigation'
import InterviewSession from '@/components/interview/InterviewSession'
import type { KbScope } from '@/components/interview/InterviewSession'
import { buildCorpus } from '@/lib/interview/corpus'
import { articleHref, findArticle } from '@/lib/knowledge'

export const dynamic = 'force-dynamic'

/**
 * 单篇过题。
 *
 * 走**路径段**而不是查询参数 —— `?article=` 那种写法离用户明确排除的
 * 「URL 位置参数」太近,而且路径更像一个可收藏的页面。
 *
 * 不选简历、不选模式:是你从某一篇文章点进来的,范围已经定死了。
 */
export default async function KbDrillPage({ params }: { params: Promise<{ title: string }> }) {
  const { title: raw } = await params
  const title = decodeURIComponent(raw)

  const corpus = buildCorpus()
  const entry = corpus.articles.find((a) => a.title === title)
  const article = findArticle(title)
  if (!entry || !article) notFound()

  // 占位稿即使手敲 URL 进来也不给考 —— 和文章页的按钮同一条规则
  const isPlaceholder = entry.state === 'placeholder'
  const questionCount = isPlaceholder
    ? 0
    : corpus.candidates.filter((c) => c.kind === 'question' && c.article === title).length
  const scope: KbScope = {
    article: title,
    chapter: entry.chapter,
    questionCount,
    examPointCount: isPlaceholder ? 0 : entry.examPoints.length,
    poolSize: questionCount + (isPlaceholder ? 0 : entry.examPoints.length),
  }

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8 flex items-baseline justify-between gap-4">
        <h1 className="truncate text-2xl font-bold">{title}</h1>
        <div className="flex shrink-0 items-baseline gap-4 text-xs">
          <Link href={articleHref(article)} className="text-gray-500 hover:text-gray-900">
            ← 回文章
          </Link>
          <Link href="/interview" className="text-gray-400 hover:text-gray-700">
            模拟面试
          </Link>
        </div>
      </div>

      {scope.poolSize === 0 ? (
        <p className="text-sm text-gray-500">
          {isPlaceholder
            ? '这一篇还是占位稿,正文都还没写 —— 没有「这一篇」可以考。写完(删掉 🚧 占位 + 补考点表)之后这里就有题了。'
            : '这一篇还没有可考的内容:既没有 topic 指向它的题库题,文末也没有「面试考点串联」表。补上考点表(见 docs/05-知识库写作契约.md 第九节)之后这里就有题了。'}
        </p>
      ) : (
        // 简历列表传空数组:单篇模式下选简历那一节根本不渲染
        <InterviewSession resumes={[]} scope={scope} />
      )}
    </main>
  )
}
