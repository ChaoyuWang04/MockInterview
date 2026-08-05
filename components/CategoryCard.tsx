import Link from 'next/link'
import type { CategoryStat } from '@/lib/types'

export default function CategoryCard({ stat }: { stat: CategoryStat }) {
  const pct = stat.total === 0 ? 0 : Math.round((stat.mastered / stat.total) * 100)
  return (
    <Link
      href={`/${encodeURIComponent(stat.name)}`}
      className="block border border-gray-200 bg-white p-5 transition-colors hover:border-gray-400"
    >
      <div className="flex items-baseline justify-between">
        <span className="font-semibold">{stat.name}</span>
        <span className="font-mono text-sm text-gray-400">{stat.total} 题</span>
      </div>
      <div className="mt-4 h-1.5 w-full bg-gray-100">
        <div className="h-full bg-blue-600" style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-2 flex justify-between text-xs">
        <span className="text-green-700">{stat.mastered} 已掌握</span>
        <span className="font-mono text-gray-400">{pct}%</span>
      </div>
    </Link>
  )
}
