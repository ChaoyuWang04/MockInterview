'use client'

import { useEffect, useMemo, useRef } from 'react'
import { groupByTopic } from '@/lib/sorting'
import type { Question } from '@/lib/types'

interface Props {
  questions: Question[]
  index: number
  onSelect: (i: number) => void
}

export default function QuestionNav({ questions, index, onSelect }: Props) {
  const activeRef = useRef<HTMLButtonElement | null>(null)

  // 与列表视图共用同一套分组逻辑(topic 第一段)
  const groups = useMemo(() => groupByTopic(questions, (q) => q.meta.topic), [questions])

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
          <div key={g.name} className="mb-3 last:mb-0">
            <div className="mb-1.5 border-b border-dashed border-gray-200 pb-1 font-mono text-[10px] tracking-widest text-gray-500">
              {g.name}
            </div>
            <div className="grid grid-cols-5 gap-1">
              {g.items.map(({ item, index: i }) => {
                const active = i === index
                return (
                  <button
                    key={item.file}
                    ref={active ? activeRef : undefined}
                    onClick={() => onSelect(i)}
                    title={`${item.meta.highfreq ? '[高频] ' : ''}${item.meta.summary ?? item.file}`}
                    className={[
                      'relative h-7 font-mono text-xs transition-colors',
                      item.meta.mastered
                        ? 'bg-green-600 text-white hover:bg-green-700'
                        : 'bg-gray-50 text-gray-600 hover:bg-gray-200',
                      active ? 'outline-2 outline-offset-1 outline-blue-600' : '',
                    ].join(' ')}
                  >
                    {i + 1}
                    {item.meta.highfreq && (
                      <span className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-red-500" />
                    )}
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
