import CategoryCard from '@/components/CategoryCard'
import { getStats } from '@/lib/questions'

export const dynamic = 'force-dynamic'

export default function Home() {
  const stats = getStats()
  return (
    <main className="mx-auto max-w-5xl px-6 py-16">
      <h1 className="text-2xl font-bold">大模型面试刷题</h1>
      <div className="mt-10">
        <div className="font-mono text-6xl font-bold text-green-700">
          {stats.mastered} <span className="text-gray-300">/</span> {stats.total}
        </div>
        <p className="mt-2 text-sm text-gray-500">已掌握 / 总题数</p>
      </div>
      <div className="mt-16 mb-6 flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">题库分类</h2>
        <span className="font-mono text-xs text-gray-400">
          {stats.categories.length} 个分类 · {stats.total} 题
        </span>
      </div>
      {stats.categories.length === 0 ? (
        <p className="text-gray-500">
          questions/ 下还没有题目。复制 questions/_template.md 到新建的分类文件夹开始出题,规范见 docs/question-authoring.md。
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
