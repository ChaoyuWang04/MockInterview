'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import type { OsPage } from '@/lib/opensource'
import Markdown from './Markdown'

interface Props {
  topic: string
  project: string
  pages: OsPage[]
}

// 阅读位置存在 sessionStorage 而不是 URL:铁律 7 排除了 URL 位置参数
const posKey = (topic: string, project: string) => `os-reader:${topic}/${project}`

export default function ProjectReader({ topic, project, pages }: Props) {
  const [index, setIndex] = useState(0)
  const [toc, setToc] = useState<{ id: string; text: string }[]>([])
  const [restored, setRestored] = useState(false)
  const articleRef = useRef<HTMLElement | null>(null)
  const pendingScroll = useRef<number | null>(null)
  const total = pages.length
  const page = pages[index]

  // 挂载时恢复上次读到的节与滚动位置
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(posKey(topic, project))
      if (raw) {
        const saved = JSON.parse(raw) as { index?: number; scrollY?: number }
        if (typeof saved.index === 'number' && saved.index >= 0 && saved.index < pages.length) {
          setIndex(saved.index)
        }
        if (typeof saved.scrollY === 'number') pendingScroll.current = saved.scrollY
      }
    } catch {
      // sessionStorage 不可用时直接从头读,不影响正文
    }
    setRestored(true)
  }, [topic, project, pages.length])

  // 正文渲染后再滚到上次的位置,并从标题生成本页目录
  useEffect(() => {
    if (!restored) return
    const el = articleRef.current
    if (!el) return
    const heads = Array.from(el.querySelectorAll('h2'))
    heads.forEach((h, i) => {
      h.id = h.id || `sec-${i}`
    })
    setToc(heads.map((h) => ({ id: h.id, text: h.textContent ?? '' })))
    if (pendingScroll.current !== null) {
      const y = pendingScroll.current
      pendingScroll.current = null
      requestAnimationFrame(() => window.scrollTo({ top: y }))
    }
  }, [restored, index, page])

  // 滚动与切节都记进 sessionStorage
  useEffect(() => {
    if (!restored) return
    let timer: ReturnType<typeof setTimeout> | null = null
    const save = () => {
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => {
        try {
          sessionStorage.setItem(
            posKey(topic, project),
            JSON.stringify({ index, scrollY: window.scrollY }),
          )
        } catch {
          // 忽略:存不进去不影响阅读
        }
      }, 200)
    }
    save()
    window.addEventListener('scroll', save, { passive: true })
    return () => {
      if (timer) clearTimeout(timer)
      window.removeEventListener('scroll', save)
    }
  }, [restored, index, topic, project])

  const go = useCallback(
    (delta: number) => {
      setIndex((i) => Math.min(total - 1, Math.max(0, i + delta)))
      window.scrollTo({ top: 0 })
    },
    [total],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.isContentEditable)) return
      if (e.key === 'ArrowRight') go(1)
      else if (e.key === 'ArrowLeft') go(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go])

  if (total === 0)
    return (
      <main className="mx-auto max-w-4xl px-6 py-16">
        <Link href="/opensource" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回解读列表
        </Link>
        <p className="mt-10 text-gray-500">这个项目还没有解读页。</p>
      </main>
    )

  return (
    <main className="mx-auto max-w-[1600px] px-6 py-10 xl:pr-56">
      <nav className="fixed right-4 top-32 hidden w-48 border border-gray-200 bg-white xl:block">
        <div className="border-b border-gray-200 px-3 py-2 font-mono text-[10px] tracking-widest text-gray-400">
          {project} · 章节
        </div>
        <div className="max-h-[60vh] overflow-y-auto p-2">
          {pages.map((p, i) => (
            <button
              key={p.file}
              onClick={() => {
                setIndex(i)
                window.scrollTo({ top: 0 })
              }}
              className={[
                'block w-full px-2 py-1.5 text-left text-sm transition-colors',
                i === index ? 'bg-gray-900 text-white' : 'text-gray-600 hover:bg-gray-100',
              ].join(' ')}
            >
              <span className="mr-1.5 font-mono text-xs opacity-60">{i}</span>
              {p.title}
            </button>
          ))}
        </div>
      </nav>

      <div className="mb-8 flex items-center gap-4 text-sm text-gray-500">
        <Link href="/opensource" className="hover:text-gray-900">
          ← 返回解读列表
        </Link>
        <span className="text-gray-300">|</span>
        <Link href="/" className="hover:text-gray-900">
          返回主页
        </Link>
        <span className="ml-auto font-mono text-xs tracking-widest text-gray-400">
          {topic} · {project} · {index + 1} / {total}
        </span>
      </div>

      {toc.length > 1 && (
        <nav className="mb-6 flex flex-wrap items-center gap-2 border border-gray-200 bg-white px-4 py-3">
          <span className="mr-1 font-mono text-[10px] tracking-widest text-gray-400">本页</span>
          {toc.map((t) => (
            <button
              key={t.id}
              onClick={() => document.getElementById(t.id)?.scrollIntoView({ block: 'start' })}
              className="border border-gray-200 px-2.5 py-1 text-sm text-gray-600 transition-colors hover:border-gray-900 hover:text-gray-900"
            >
              {t.text}
            </button>
          ))}
        </nav>
      )}

      <article ref={articleRef} className="border border-gray-200 bg-white p-8">
        <div className="prose os-prose max-w-none">
          <Markdown>{page.content}</Markdown>
        </div>
      </article>

      <div className="mt-6 flex items-center justify-between">
        <button
          onClick={() => go(-1)}
          disabled={index === 0}
          className="border border-gray-300 bg-white px-4 py-2 text-sm disabled:opacity-40"
        >
          ← 上一节{index > 0 ? `:${pages[index - 1].title}` : ''}
        </button>
        <span className="font-mono text-xs text-gray-400">←/→ 翻节</span>
        <button
          onClick={() => go(1)}
          disabled={index === total - 1}
          className="border border-gray-300 bg-white px-4 py-2 text-sm disabled:opacity-40"
        >
          {index < total - 1 ? `下一节:${pages[index + 1].title}` : '已是最后一节'} →
        </button>
      </div>
    </main>
  )
}
