import Link from 'next/link'
import { countOsPages, listOsProjects, listOsTopics } from '@/lib/opensource'

export const dynamic = 'force-dynamic'

export default function OsIndexPage() {
  const topics = listOsTopics().map((t) => ({
    name: t,
    projects: listOsProjects(t).map((p) => ({ name: p, pages: countOsPages(t, p) })),
  }))
  const total = topics.reduce((s, t) => s + t.projects.length, 0)
  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">开源项目解读</h1>
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
      </div>
      <p className="mt-2 text-sm text-gray-500">
        每个项目 = 一组按子系统分页的解读文档:第 0 节是全景总览,之后每节拆一个子系统。共 {total} 个项目已解读;完整项目清单与排期见 docs/05-开源解读流程.md。
      </p>
      {total === 0 ? (
        <p className="mt-10 text-gray-500">
          还没有解读。源码仓在 projects/&lt;主题&gt;/,解读写到 opensource/&lt;主题&gt;/&lt;项目&gt;/,规范见 docs/05-开源解读流程.md。
        </p>
      ) : (
        topics
          .filter((t) => t.projects.length > 0)
          .map((t) => (
            <section key={t.name} className="mt-10">
              <h2 className="mb-3 font-mono text-xs tracking-widest text-gray-400">{t.name}</h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {t.projects.map((p) => (
                  <Link
                    key={p.name}
                    href={`/opensource/${encodeURIComponent(t.name)}/${encodeURIComponent(p.name)}`}
                    className="block border border-gray-200 bg-white px-4 py-3 transition-colors hover:border-gray-400"
                  >
                    <span className="font-semibold">{p.name}</span>
                    <span className="ml-2 font-mono text-xs text-gray-400">{p.pages} 节</span>
                  </Link>
                ))}
              </div>
            </section>
          ))
      )}
    </main>
  )
}
