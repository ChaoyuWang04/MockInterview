import Link from 'next/link'
import { notFound } from 'next/navigation'
import Markdown from '@/components/Markdown'
import { getReport } from '@/lib/reports'

export const dynamic = 'force-dynamic'

export default async function ReportPage({
  params,
}: {
  params: Promise<{ company: string; report: string }>
}) {
  const { company, report: slug } = await params
  const report = getReport(company, slug)
  if (!report) notFound()

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <div className="mb-8 flex items-center gap-4 text-sm text-gray-500">
        <Link href="/reports" className="hover:text-gray-900">
          ← 返回报告解读
        </Link>
        <span className="text-gray-300">|</span>
        <Link href="/" className="hover:text-gray-900">
          返回主页
        </Link>
        <span className="ml-auto font-mono text-xs tracking-widest text-gray-400">
          {report.company}
        </span>
      </div>

      <article className="border border-gray-200 bg-white p-8">
        <div className="prose max-w-none">
          <Markdown>{report.content}</Markdown>
        </div>
      </article>
    </main>
  )
}
