'use client'

import { useEffect, useRef, useState } from 'react'
import type { LcGroup, LcProblem } from '@/lib/leetcode'

const DIFFICULTY_STYLE: Record<string, string> = {
  简单: 'text-green-600',
  中等: 'text-amber-600',
  困难: 'text-red-600',
}

interface Props {
  groups: LcGroup[]
  initialNotes: Record<string, string>
}

export default function Hot100List({ groups, initialNotes }: Props) {
  const [notes, setNotes] = useState(initialNotes)
  const [open, setOpen] = useState<LcProblem | null>(null)
  const [draft, setDraft] = useState('')
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const total = groups.reduce((s, g) => s + g.problems.length, 0)
  const noteCount = Object.keys(notes).length

  const openPanel = (p: LcProblem) => {
    setOpen(p)
    setDraft(notes[p.slug] ?? '')
    setStatus('idle')
  }

  const save = async (slug: string, value: string) => {
    setStatus('saving')
    try {
      const res = await fetch('/api/leetcode-note', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug, note: value }),
      })
      if (!res.ok) throw new Error('save failed')
      setNotes((n) => {
        const next = { ...n }
        if (value.trim()) next[slug] = value
        else delete next[slug]
        return next
      })
      setStatus('saved')
    } catch {
      setStatus('error')
    }
  }

  // 输入停顿 800ms 自动保存
  const onChange = (value: string) => {
    setDraft(value)
    setStatus('saving')
    if (timer.current) clearTimeout(timer.current)
    const slug = open?.slug
    if (!slug) return
    timer.current = setTimeout(() => save(slug, value), 800)
  }

  const closePanel = () => {
    if (timer.current) clearTimeout(timer.current)
    if (open && draft !== (notes[open.slug] ?? '')) save(open.slug, draft)
    setOpen(null)
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) closePanel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <>
      <p className="mt-2 text-sm text-gray-500">
        力扣官方「热题 100」清单,按 {groups.length} 个专题组织,共 {total} 题;已记录 {noteCount} 条笔记。点 📝 写解题小技巧,面板内可直达 LeetCode / 力扣。
      </p>

      {groups.map((g) => (
        <section key={g.name} className="mt-8">
          <h2 className="mb-2 flex items-baseline gap-2 font-mono text-xs tracking-widest text-gray-400">
            {g.name}
            <span className="text-gray-300">{g.problems.length} 题</span>
          </h2>
          <div className="border border-gray-200 bg-white">
            {g.problems.map((p, i) => (
              <div
                key={p.slug}
                className={[
                  'flex items-center gap-3 px-4 py-2.5',
                  i > 0 ? 'border-t border-gray-100' : '',
                ].join(' ')}
              >
                <span className="w-12 shrink-0 font-mono text-xs text-gray-400">{p.id}</span>
                <a
                  href={`https://leetcode.cn/problems/${p.slug}/`}
                  target="_blank"
                  rel="noreferrer"
                  className="flex-1 truncate text-sm text-gray-900 hover:text-blue-600 hover:underline"
                >
                  {p.title}
                </a>
                <button
                  onClick={() => openPanel(p)}
                  title={notes[p.slug] ? '已有笔记' : '写笔记'}
                  className={[
                    'shrink-0 border px-2 py-0.5 text-xs transition-colors',
                    notes[p.slug]
                      ? 'border-blue-300 bg-blue-50 text-blue-600'
                      : 'border-gray-200 text-gray-400 hover:border-gray-400 hover:text-gray-700',
                  ].join(' ')}
                >
                  📝
                </button>
                <span
                  className={`w-10 shrink-0 text-right text-xs ${DIFFICULTY_STYLE[p.difficulty] ?? 'text-gray-500'}`}
                >
                  {p.difficulty}
                </span>
              </div>
            ))}
          </div>
        </section>
      ))}

      {open && (
        <aside className="fixed bottom-6 right-6 z-50 w-80 border border-gray-300 bg-white shadow-xl">
          <div className="flex items-center gap-2 border-b border-gray-200 px-3 py-2">
            <span className="flex-1 truncate text-sm font-semibold">
              <span className="mr-1.5 font-mono text-xs text-gray-400">{open.id}</span>
              {open.title}
            </span>
            <button
              onClick={closePanel}
              className="shrink-0 px-1 text-lg leading-none text-gray-400 hover:text-gray-900"
              title="关闭(Esc)"
            >
              ×
            </button>
          </div>
          <textarea
            value={draft}
            onChange={(e) => onChange(e.target.value)}
            autoFocus
            rows={8}
            placeholder="解题小技巧、易错点…(自动保存)"
            className="w-full resize-y border-0 p-3 font-mono text-xs leading-relaxed focus:outline-none"
          />
          <div className="flex items-center gap-2 border-t border-gray-200 px-3 py-2">
            <a
              href={`https://leetcode.com/problems/${open.slug}/`}
              target="_blank"
              rel="noreferrer"
              className="border border-gray-300 px-2 py-1 text-xs text-gray-700 hover:border-gray-500"
            >
              LeetCode US ↗
            </a>
            <a
              href={`https://leetcode.cn/problems/${open.slug}/`}
              target="_blank"
              rel="noreferrer"
              className="border border-gray-300 px-2 py-1 text-xs text-gray-700 hover:border-gray-500"
            >
              力扣中国 ↗
            </a>
            <span className="ml-auto font-mono text-[10px] text-gray-400">
              {status === 'saving' ? '保存中…' : status === 'saved' ? '已保存' : status === 'error' ? '保存失败' : ''}
            </span>
          </div>
        </aside>
      )}
    </>
  )
}
