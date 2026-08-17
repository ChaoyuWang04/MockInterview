import Link from 'next/link'
import Hot100List from '@/components/Hot100List'
import { getAllNotes, listHighFreq, listHot100 } from '@/lib/leetcode'

export const dynamic = 'force-dynamic'

export default function LeetcodePage() {
  const groups = listHot100()
  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">LeetCode 热题 100</h1>
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
      </div>
      {groups.length === 0 ? (
        <p className="mt-10 text-gray-500">
          清单为空。数据文件是 leetcode/hot100.md,格式与校准方式见 docs/leetcode-hot100.md。
        </p>
      ) : (
        <Hot100List groups={groups} initialNotes={getAllNotes()} initialHighFreq={listHighFreq()} />
      )}
    </main>
  )
}
