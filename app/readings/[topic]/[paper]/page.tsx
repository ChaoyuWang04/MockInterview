import type { Metadata } from 'next'
import Link from 'next/link'
import { notFound } from 'next/navigation'
import ArticleFloatNav from '@/components/ArticleFloatNav'
import Markdown from '@/components/Markdown'
import { originalPdfHref } from '@/lib/original-pdf'
import { getReading } from '@/lib/readings'

export const dynamic = 'force-dynamic'

export async function generateMetadata({
  params,
}: {
  params: Promise<{ topic: string; paper: string }>
}): Promise<Metadata> {
  const { paper: rawSlug } = await params
  return { title: decodeURIComponent(rawSlug) }
}

export default async function ReadingPage({
  params,
}: {
  params: Promise<{ topic: string; paper: string }>
}) {
  // 动态段是百分号编码送进来的,方向名全是中文,必须解码;/opensource 同理。
  // /reports 不解是因为公司名都是 ASCII,解与不解等价——那条断言不适用于本路由。
  const { topic: rawTopic, paper: rawSlug } = await params
  const topic = decodeURIComponent(rawTopic)
  const slug = decodeURIComponent(rawSlug)
  const reading = getReading(topic, slug)
  if (!reading) notFound()
  const pdfHref = originalPdfHref({ lib: 'readings', topic, slug })

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <div className="mb-8 flex items-center gap-4 text-sm text-gray-500">
        <Link href="/readings" className="hover:text-gray-900">
          ← 返回日常研读
        </Link>
        <span className="text-gray-300">|</span>
        <Link href="/" className="hover:text-gray-900">
          返回主页
        </Link>
        {pdfHref ? (
          <>
            <span className="text-gray-300">|</span>
            <a
              href={pdfHref}
              target="_blank"
              rel="noreferrer"
              className="font-mono text-[10px] tracking-widest text-gray-400 hover:text-gray-900"
            >
              原文
            </a>
          </>
        ) : null}
        <span className="ml-auto font-mono text-xs tracking-widest text-gray-400">
          {reading.topic}
          {reading.org ? ` · ${reading.org}` : ''}
        </span>
      </div>

      <article className="border border-gray-200 bg-white p-8">
        <div className="prose max-w-none">
          <Markdown>{reading.content}</Markdown>
        </div>
      </article>
      <ArticleFloatNav />
    </main>
  )
}
