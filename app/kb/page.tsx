import Link from 'next/link'
import { articleHref, countArticles, flattenArticles, listKbTree, type KbFolder } from '@/lib/knowledge'

export const dynamic = 'force-dynamic'

function ArticleGrid({ folder }: { folder: KbFolder }) {
  if (folder.articles.length === 0) return null
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {folder.articles.map((a) => (
        <Link
          key={a.segments.join('/')}
          href={articleHref(a)}
          className="flex items-center justify-between gap-2 border border-gray-200 bg-white px-4 py-3 font-semibold transition-colors hover:border-gray-400"
        >
          <span className="truncate">{a.title}</span>
        </Link>
      ))}
    </div>
  )
}

/** 递归渲染:一级章 = 大标题,二级子领域 = 小标题 */
function Section({ folder, depth }: { folder: KbFolder; depth: number }) {
  const total = countArticles(folder)
  return (
    <section className={depth === 0 ? 'mt-12' : 'mt-6'}>
      {depth === 0 ? (
        <h2 className="mb-3 flex items-baseline gap-2 border-b border-gray-200 pb-1 text-lg font-bold">
          {folder.title}
          <span className="font-mono text-xs font-normal text-gray-400">{total} 篇</span>
        </h2>
      ) : (
        <h3 className="mb-2 flex items-baseline gap-2 font-mono text-xs tracking-widest text-gray-400">
          {folder.title}
          <span className="text-gray-300">{total} 篇</span>
        </h3>
      )}
      <ArticleGrid folder={folder} />
      {folder.folders.map((f) => (
        <Section key={f.segments.join('/')} folder={f} depth={depth + 1} />
      ))}
    </section>
  )
}

export default function KbIndexPage() {
  const tree = listKbTree()
  const articles = flattenArticles(tree)
  const total = articles.length

  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">知识库</h1>
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
      </div>
      <p className="mt-2 text-sm text-gray-500">
        按训练流程编排:模型结构 → 预训练与微调 → 强化学习 → Infra → 多模态 → 应用。共 {total} 篇。
      </p>
      {total === 0 ? (
        <p className="mt-10 text-gray-500">还没有文章。规范见 docs/04-知识库地图.md。</p>
      ) : (
        <>
          <ArticleGrid folder={tree} />
          {tree.folders.map((f) => (
            <Section key={f.segments.join('/')} folder={f} depth={0} />
          ))}
        </>
      )}
    </main>
  )
}
