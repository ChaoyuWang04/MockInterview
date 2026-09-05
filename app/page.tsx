import Link from 'next/link'
import CategoryCard from '@/components/CategoryCard'
import { countArticles, listKbTree } from '@/lib/knowledge'
import { listHot100 } from '@/lib/leetcode'
import { listSessions } from '@/lib/interview/session'
import { listOsProjects, listOsTopics } from '@/lib/opensource'
import { getStats } from '@/lib/questions'
import { countReports } from '@/lib/reports'

export const dynamic = 'force-dynamic'

export default function Home() {
  const stats = getStats()
  const kbCount = countArticles(listKbTree())
  const osCount = listOsTopics().reduce((s, t) => s + listOsProjects(t).length, 0)
  const reportCount = countReports()
  const lcCount = listHot100().reduce((s, g) => s + g.problems.length, 0)
  const sessionCount = listSessions().length
  return (
    <main className="mx-auto max-w-5xl px-6 py-16">
      <h1 className="text-2xl font-bold">大模型面试刷题</h1>
      <div className="mt-10">
        <div className="font-mono text-6xl font-bold text-green-700">
          {stats.mastered} <span className="text-gray-300">/</span> {stats.total}
        </div>
        <p className="mt-2 text-sm text-gray-500">已掌握 / 总题数</p>
        <div className="mt-4 flex flex-wrap gap-3">
          <Link
            href="/interview"
            className="inline-block border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 transition-colors hover:border-gray-500"
          >
            🎤 模拟面试{sessionCount > 0 ? ` · ${sessionCount} 场` : ''} →
          </Link>
          <Link
            href="/kb"
            className="inline-block border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 transition-colors hover:border-gray-500"
          >
            📚 知识库 · {kbCount} 篇 →
          </Link>
          <Link
            href="/opensource"
            className="inline-block border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 transition-colors hover:border-gray-500"
          >
            🔍 开源项目解读 · {osCount} 个 →
          </Link>
          <Link
            href="/reports"
            className="inline-block border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 transition-colors hover:border-gray-500"
          >
            📄 报告解读 · {reportCount} 篇 →
          </Link>
          <Link
            href="/leetcode"
            className="inline-block border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 transition-colors hover:border-gray-500"
          >
            💯 LeetCode 热题 · {lcCount} 题 →
          </Link>
        </div>
      </div>
      <div className="mt-16 mb-6 flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">题库分类</h2>
        <span className="font-mono text-xs text-gray-400">
          {stats.categories.length} 个分类 · {stats.total} 题
        </span>
      </div>
      {stats.categories.length === 0 ? (
        <p className="text-gray-500">
          questions/ 下还没有题目。复制 questions/_template.md 到新建的分类文件夹开始出题,规范见 docs/03-题目写作规范.md。
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {stats.categories.map((c) => (
            <CategoryCard key={c.name} stat={c} />
          ))}
        </div>
      )}
    </main>
  )
}
