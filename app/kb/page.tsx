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
          className={[
            'flex items-center justify-between gap-2 border bg-white px-4 py-3 transition-colors',
            a.placeholder
              ? 'border-dashed border-gray-200 text-gray-400 hover:border-gray-400'
              : 'border-gray-200 font-semibold hover:border-gray-400',
          ].join(' ')}
        >
          <span className="truncate">{a.title}</span>
          {a.placeholder && <span className="shrink-0 font-mono text-[10px]">占位</span>}
          {a.legacy && (
            <span
              title="写作契约确立前的旧稿,待按新标准重写"
              className="shrink-0 border border-amber-300 bg-amber-50 px-1 font-mono text-[10px] font-normal text-amber-600"
            >
              旧
            </span>
          )}
          {a.keypoint && (
            <span
              title="重点复习文章,文末有「面试考点串联」"
              className="shrink-0 bg-red-600 px-1 font-mono text-[10px] font-bold text-white"
            >
              重
            </span>
          )}
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
  const placeholders = articles.filter((a) => a.placeholder).length
  const legacy = articles.filter((a) => a.legacy).length
  const keypoints = articles.filter((a) => a.keypoint).length

  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">知识库</h1>
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
      </div>
      <p className="mt-2 text-sm text-gray-500">
        按训练流程编排:模型结构 → 预训练与微调 → 强化学习 → Infra → 多模态 → 应用。
        共 {total} 篇:{total - placeholders - legacy} 篇已按写作契约成文、
        <span className="text-amber-600">{legacy} 篇旧稿待重写(标「旧」)</span>、{placeholders} 篇占位(虚线框)。
        <span className="text-red-600">红「重」= 重点复习文章({keypoints} 篇)</span>。
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
