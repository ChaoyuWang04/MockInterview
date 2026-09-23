'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'

/** 两个解读库索引页共用的卡片;服务端把「按方向」与「按时间」两种分组都算好,这里只切换与打已读 */
export interface LibraryCard {
  /** `<目录>/<材料>`,已读标记与票数都按它对应 */
  key: string
  href: string
  title: string
  badge: string
  topic: string
  releaseDate: string
  votes: number | null
}

export interface LibraryGroup {
  title: string
  cards: LibraryCard[]
}

type View = 'topic' | 'time'

const VIEWS: { value: View; label: string }[] = [
  { value: 'topic', label: '按方向' },
  { value: 'time', label: '按时间' },
]

export default function LibraryIndex({
  library,
  badgeTitle,
  byTopic,
  byTime,
  initialRead,
}: {
  library: 'reports' | 'readings'
  badgeTitle: string
  byTopic: LibraryGroup[]
  byTime: LibraryGroup[]
  initialRead: string[]
}) {
  const storageKey = `library-view:${library}`
  const [view, setView] = useState<View>('topic')
  const [read, setRead] = useState(() => new Set(initialRead))
  const [error, setError] = useState<string | null>(null)

  // 视图偏好只是本机的浏览习惯,不进 markdown,也不进 URL(URL 位置参数是明确排除的功能)
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(storageKey)
      if (saved === 'topic' || saved === 'time') setView(saved)
    } catch {}
  }, [storageKey])

  function choose(next: View) {
    setView(next)
    try {
      window.localStorage.setItem(storageKey, next)
    } catch {}
  }

  async function toggleRead(key: string) {
    const next = !read.has(key)
    const apply = (value: boolean) =>
      setRead((current) => {
        const copy = new Set(current)
        if (value) copy.add(key)
        else copy.delete(key)
        return copy
      })
    apply(next)
    setError(null)
    try {
      const res = await fetch('/api/read-mark', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ library, key, read: next }),
      })
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error ?? res.statusText)
    } catch (e) {
      apply(!next)
      setError(`已读标记没有保存:${(e as Error).message}`)
    }
  }

  const groups = view === 'topic' ? byTopic : byTime

  return (
    <>
      <div className="mt-6 flex flex-wrap items-center gap-3">
        <div role="radiogroup" aria-label="排序方式" className="inline-flex border border-gray-300 bg-white text-sm">
          {VIEWS.map((option) => (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={view === option.value}
              onClick={() => choose(option.value)}
              className={`px-3 py-1 transition-colors ${
                view === option.value ? 'bg-gray-900 text-white' : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
        <span className="text-xs text-gray-400">
          {view === 'time'
            ? '按首发日距今分段;段内按 alphaXiv 达标当天的票数从高到低,无票数的排后'
            : '每个方向按首发日从新到旧'}
          {' · '}已读 {read.size} 篇
        </span>
      </div>
      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

      {groups.map((group) => (
        <section key={group.title} className="mt-10">
          <h2 className="mb-3 flex items-baseline gap-2 border-b border-gray-200 pb-1 text-lg font-bold">
            {group.title}
            <span className="font-mono text-xs font-normal text-gray-400">{group.cards.length} 篇</span>
          </h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {group.cards.map((card) => {
              const isRead = read.has(card.key)
              return (
                <div
                  key={card.key}
                  className={`relative border border-gray-200 bg-white transition-colors hover:border-gray-400 ${
                    isRead ? 'opacity-55' : ''
                  }`}
                >
                  <Link href={card.href} className="block px-5 pt-4 pb-10">
                    <span className="flex items-start justify-between gap-3">
                      <span className="font-semibold">{card.title}</span>
                      <span
                        title={badgeTitle}
                        className="mt-0.5 shrink-0 border border-gray-300 bg-gray-50 px-1 font-mono text-[10px] font-normal text-gray-500"
                      >
                        {card.badge}
                      </span>
                    </span>
                  </Link>
                  <div className="pointer-events-none absolute inset-x-5 bottom-3 flex items-center justify-between gap-3 font-mono text-xs text-gray-400">
                    <span className="min-w-0 truncate">
                      {card.releaseDate} 首发
                      {card.votes !== null && (
                        <span title="alphaXiv 达标当天的票数" className="ml-2 text-gray-500">
                          ▲ {card.votes}
                        </span>
                      )}
                      {view === 'time' && <span className="ml-2">· {card.topic}</span>}
                    </span>
                    <label className="pointer-events-auto flex shrink-0 cursor-pointer items-center gap-1 select-none hover:text-gray-700">
                      <input
                        type="checkbox"
                        checked={isRead}
                        onChange={() => toggleRead(card.key)}
                        className="size-3.5 cursor-pointer accent-gray-700"
                      />
                      已读
                    </label>
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      ))}
    </>
  )
}
