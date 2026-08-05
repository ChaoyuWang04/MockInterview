'use client'

import { useEffect, useMemo, useRef } from 'react'
import type { Question } from '@/lib/types'

interface Props {
  questions: Question[]
  index: number
  onSelect: (i: number) => void
}

export default function QuestionNav({ questions, index, onSelect }: Props) {
  const activeRef = useRef<HTMLButtonElement | null>(null)

  const groups = useMemo(() => {
    const list: { head: string; items: { i: number; mastered: boolean }[] }[] = []
    const byHead = new Map<string, number>()
    questions.forEach((q, i) => {
      const head = q.meta.topic?.split('/')[0]?.trim() || '未分层'
      let gi = byHead.get(head)
      if (gi === undefined) {
        gi = list.length
        byHead.set(head, gi)
        list.push({ head, items: [] })
      }
      list[gi].items.push({ i, mastered: q.meta.mastered })
    })
    return list
  }, [questions])

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest' })
  }, [index])

  return (
    <nav className="fixed right-4 top-32 hidden w-44 border border-gray-200 bg-white xl:block">
      <div className="border-b border-gray-200 px-3 py-2 font-mono text-[10px] tracking-widest text-gray-400">
        题目导航
      </div>
      <div className="max-h-[56vh] overflow-y-auto p-3">
        {groups.map((g) => (
          <div key={g.head} className="mb-3 last:mb-0">
            <div className="mb-1.5 border-b border-dashed border-gray-200 pb-1 font-mono text-[10px] tracking-widest text-gray-500">
              {g.head}
            </div>
            <div className="grid grid-cols-5 gap-1">
              {g.items.map(({ i, mastered }) => {
                const active = i === index
                return (
                  <button
                    key={i}
                    ref={active ? activeRef : undefined}
                    onClick={() => onSelect(i)}
                    title={questions[i].meta.summary ?? ''}
                    className={[
                      'h-7 font-mono text-xs transition-colors',
                      mastered
                        ? 'bg-green-600 text-white hover:bg-green-700'
                        : 'bg-gray-50 text-gray-600 hover:bg-gray-200',
                      active ? 'outline-2 outline-offset-1 outline-blue-600' : '',
                    ].join(' ')}
                  >
                    {i + 1}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </nav>
  )
}
