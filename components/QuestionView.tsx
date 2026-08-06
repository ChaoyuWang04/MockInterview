'use client'

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import type { Question, SectionName } from '@/lib/types'
import Markdown from './Markdown'
import NoteEditor from './NoteEditor'
import QuestionNav from './QuestionNav'

const ANSWER_SECTIONS: { name: SectionName; label: string }[] = [
  { name: '要点', label: 'KEY POINTS' },
  { name: '答案', label: 'REFERENCE ANSWER' },
  { name: '知识点', label: 'KNOWLEDGE POINT' },
  { name: '追问', label: '追问题目' },
]

const DIFFICULTY_STYLE: Record<string, string> = {
  简单: 'text-green-600',
  中等: 'text-amber-600',
  困难: 'text-red-600',
}

interface Props {
  category: string
  initialQuestions: Question[]
  kbTopics?: string[]
}

export default function QuestionView({ category, initialQuestions, kbTopics = [] }: Props) {
  const [questions, setQuestions] = useState(initialQuestions)
  const [index, setIndex] = useState(0)
  const [expanded, setExpanded] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const total = questions.length
  const masteredCount = questions.filter((q) => q.meta.mastered).length
  const q = questions[index]
  const topicHead = q?.meta.topic?.split('/')[0]?.trim()
  const kbTopic = topicHead && kbTopics.includes(topicHead) ? topicHead : null

  const go = useCallback(
    (delta: number) => {
      setIndex((i) => Math.min(total - 1, Math.max(0, i + delta)))
      setExpanded(false)
    },
    [total],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.isContentEditable)) return
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault()
        setExpanded((v) => !v)
      } else if (e.key === 'ArrowRight') go(1)
      else if (e.key === 'ArrowLeft') go(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go])

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 2500)
  }

  const patch = async (payload: { mastered?: boolean; note?: string }) => {
    try {
      const res = await fetch('/api/question', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category, file: q.file, ...payload }),
      })
      return res.ok
    } catch {
      return false
    }
  }

  const toggleMastered = async () => {
    const next = !q.meta.mastered
    if (await patch({ mastered: next })) {
      setQuestions((qs) =>
        qs.map((item, i) => (i === index ? { ...item, meta: { ...item.meta, mastered: next } } : item)),
      )
    } else showToast('保存失败,请重试')
  }

  const saveNote = async (note: string) => {
    const ok = await patch({ note })
    if (ok)
      setQuestions((qs) =>
        qs.map((item, i) => (i === index ? { ...item, sections: { ...item.sections, Note: note } } : item)),
      )
    else showToast('Note 保存失败,请重试')
    return ok
  }

  if (total === 0)
    return (
      <main className="mx-auto max-w-4xl px-6 py-16">
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
        <p className="mt-10 text-gray-500">
          这个分类还没有题目。复制 questions/_template.md 到 questions/{category}/ 开始出题。
        </p>
      </main>
    )

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <QuestionNav
        questions={questions}
        index={index}
        onSelect={(i) => {
          setIndex(i)
          setExpanded(false)
        }}
      />
      <div className="mb-8 flex items-start justify-between">
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
          ← 返回主页
        </Link>
        <div className="border border-green-700 px-4 py-2 text-right">
          <div className="font-mono text-xs tracking-widest text-gray-400">{category} 掌握</div>
          <div className="font-mono text-lg text-green-700">
            {masteredCount} / {total}
          </div>
          <div className="mt-1 h-1 w-32 bg-gray-100">
            <div
              className="h-full bg-green-600"
              style={{ width: `${total ? (masteredCount / total) * 100 : 0}%` }}
            />
          </div>
        </div>
      </div>

      <article className="border border-gray-200 bg-white p-8">
        {q.error ? (
          <div className="border border-red-300 bg-red-50 p-4 text-sm text-red-700">
            <p className="font-mono">
              {category}/{q.file}
            </p>
            <p className="mt-1">{q.error}</p>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3 font-mono text-xs tracking-widest">
              <span className="text-blue-600">{category}</span>
              {q.meta.difficulty && (
                <span className={DIFFICULTY_STYLE[q.meta.difficulty] ?? 'text-gray-500'}>
                  {q.meta.difficulty}
                </span>
              )}
              <span className="text-gray-400">
                #{index + 1} / {total}
              </span>
            </div>

            <div className="prose mt-4 max-w-none font-semibold">
              <Markdown>{q.sections['题目'] ?? ''}</Markdown>
            </div>

            {(q.meta.tags.length > 0 || q.meta.company || kbTopic) && (
              <div className="mt-4 flex flex-wrap gap-2">
                {q.meta.tags.map((t) => (
                  <span key={t} className="border border-gray-200 px-2 py-0.5 text-xs text-gray-600">
                    {t}
                  </span>
                ))}
                {q.meta.company && (
                  <span className="border border-gray-100 bg-gray-50 px-2 py-0.5 text-xs text-gray-400">
                    {q.meta.company}
                  </span>
                )}
                {kbTopic && (
                  <Link
                    href={`/kb/${encodeURIComponent(category)}/${encodeURIComponent(kbTopic)}`}
                    className="border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs text-blue-600 hover:border-blue-400"
                  >
                    📚 知识库:{kbTopic}
                  </Link>
                )}
              </div>
            )}

            <div className="mt-8 text-center">
              <button onClick={() => setExpanded((v) => !v)} className="text-blue-600 hover:underline">
                {expanded ? '收起答案' : '展开答案'}
              </button>
              <p className="mt-1 font-mono text-[10px] tracking-widest text-gray-400">SPACE 快捷键</p>
            </div>

            {expanded &&
              ANSWER_SECTIONS.map(({ name, label }) =>
                q.sections[name] ? (
                  <section key={name} className="mt-8 border-t border-dashed border-gray-200 pt-6">
                    <h3 className="mb-3 font-mono text-xs tracking-widest text-gray-400">{label}</h3>
                    <div className="prose prose-sm max-w-none">
                      <Markdown>{q.sections[name]!}</Markdown>
                    </div>
                  </section>
                ) : null,
              )}

            <div className="mt-10 border-t border-gray-200 pt-6">
              <button
                onClick={toggleMastered}
                className={
                  q.meta.mastered
                    ? 'border border-green-700 bg-green-700 px-4 py-2 text-sm text-white'
                    : 'border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:border-green-700 hover:text-green-700'
                }
              >
                {q.meta.mastered ? '✓ 已掌握' : '标记已掌握'}
              </button>
            </div>

            <NoteEditor key={`${category}/${q.file}`} note={q.sections['Note'] ?? ''} onSave={saveNote} />
          </>
        )}
      </article>

      <div className="mt-6 flex items-center justify-between">
        <button
          onClick={() => go(-1)}
          disabled={index === 0}
          className="border border-gray-300 bg-white px-4 py-2 text-sm disabled:opacity-40"
        >
          ← 上一题
        </button>
        <span className="font-mono text-xs text-gray-400">←/→ 切题 · 空格 展开</span>
        <button
          onClick={() => go(1)}
          disabled={index === total - 1}
          className="border border-gray-300 bg-white px-4 py-2 text-sm disabled:opacity-40"
        >
          下一题 →
        </button>
      </div>

      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-gray-900 px-4 py-2 text-sm text-white">
          {toast}
        </div>
      )}
    </main>
  )
}
