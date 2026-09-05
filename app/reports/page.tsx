import Link from 'next/link'
import { listReportCompanies, listReports } from '@/lib/reports'

export const dynamic = 'force-dynamic'

export default function ReportsIndexPage() {
  const companies = listReportCompanies().map((company) => ({
    name: company,
    reports: listReports(company),
  }))
  const total = companies.reduce((sum, company) => sum + company.reports.length, 0)

  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">报告解读</h1>
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
      </div>
      <p className="mt-2 text-sm leading-6 text-gray-500">
        按公司整理公开的基模 Technical Report
        与核心技术论文。每篇用一篇完整长文讲清设计、原因和可迁移的部分。当前共 {total} 篇。
      </p>

      {total === 0 ? (
        <div className="mt-10 border border-dashed border-gray-300 bg-white p-6 text-sm leading-7 text-gray-500">
          还没有已发布的解读。PDF 原件放到 papers/&lt;公司&gt;/，完成的长文放到
          reports/&lt;公司&gt;/；完整流程见 docs/10-基模报告流程.md。
        </div>
      ) : (
        companies
          .filter((company) => company.reports.length > 0)
          .map((company) => (
            <section key={company.name} className="mt-10">
              <h2 className="mb-3 font-mono text-xs tracking-widest text-gray-400">
                {company.name}
              </h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {company.reports.map((report) => (
                  <Link
                    key={report.slug}
                    href={`/reports/${encodeURIComponent(company.name)}/${encodeURIComponent(report.slug)}`}
                    className="block border border-gray-200 bg-white px-5 py-4 transition-colors hover:border-gray-400"
                  >
                    <span className="font-semibold">{report.title}</span>
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
