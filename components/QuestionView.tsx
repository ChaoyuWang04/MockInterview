'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { groupByTopic, sortQuestions } from '@/lib/sorting'
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
  /** topic 第一段 → 知识库文章链接 */
  kbLinks?: Record<string, string>
}

export default function QuestionView({ category, initialQuestions, kbLinks = {} }: Props) {
  // 原始数组只用于承载数据;列表与刷题都走排序后的 questions(高频前置 + 桶内按难度)
  const [raw, setRaw] = useState(initialQuestions)
  const [mode, setMode] = useState<'list' | 'drill'>('list')
  const [index, setIndex] = useState(0)
  const [expanded, setExpanded] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  // 分组顺序按文件顺序固定(标高频不会让整个主题跳到最前),只在组内排序:
  // 高频前置 + 桶内 简单→中等→困难。列表、刷题、右侧导航共用这一份顺序。
  const groups = useMemo(() => {
    let offset = 0
    return groupByTopic(raw, (item) => item.meta.topic).map((g) => {
      const items = sortQuestions(g.items.map((x) => x.item)).map((item, i) => ({
        item,
        index: offset + i,
      }))
      offset += items.length
      return { name: g.name, items }
    })
  }, [raw])
  const questions = useMemo(() => groups.flatMap((g) => g.items.map((x) => x.item)), [groups])
  const total = questions.length
  const masteredCount = questions.filter((q) => q.meta.mastered).length
  const highFreqCount = questions.filter((q) => q.meta.highfreq).length
  const q = questions[index]
  const topicHead = q?.meta.topic?.split('/')[0]?.trim()
  const kbHref = topicHead ? kbLinks[topicHead] : undefined

  const go = useCallback(
    (delta: number) => {
      setIndex((i) => Math.min(total - 1, Math.max(0, i + delta)))
      setExpanded(false)
    },
    [total],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (mode !== 'drill') return
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.isContentEditable)) return
      if (e.key === 'Escape') {
        setMode('list')
        return
      }
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault()
        setExpanded((v) => !v)
      } else if (e.key === 'ArrowRight') go(1)
      else if (e.key === 'ArrowLeft') go(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go, mode])

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 2500)
  }

  const patch = async (
    file: string,
    payload: { mastered?: boolean; note?: string; highfreq?: boolean },
  ) => {
    try {
      const res = await fetch('/api/question', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category, file, ...payload }),
      })
      return res.ok
    } catch {
      return false
    }
  }

  /** 按 file 更新本地状态(排序会改变下标,不能用 index 定位) */
  const patchLocal = (file: string, update: (item: Question) => Question) =>
    setRaw((qs) => qs.map((item) => (item.file === file ? update(item) : item)))

  const toggleMastered = async () => {
    const next = !q.meta.mastered
    if (await patch(q.file, { mastered: next }))
      patchLocal(q.file, (item) => ({ ...item, meta: { ...item.meta, mastered: next } }))
    else showToast('保存失败,请重试')
  }

  const toggleHighFreq = async (target: Question) => {
    const next = !target.meta.highfreq
    // 乐观更新:列表会立刻重排;失败则回滚
    patchLocal(target.file, (item) => ({ ...item, meta: { ...item.meta, highfreq: next } }))
    if (!(await patch(target.file, { highfreq: next }))) {
      patchLocal(target.file, (item) => ({ ...item, meta: { ...item.meta, highfreq: !next } }))
      showToast('高频标记保存失败,请重试')
    }
  }

  const saveNote = async (note: string) => {
    const ok = await patch(q.file, { note })
    if (ok) patchLocal(q.file, (item) => ({ ...item, sections: { ...item.sections, Note: note } }))
    else showToast('Note 保存失败,请重试')
    return ok
  }

  const openDrill = (i: number) => {
    setIndex(i)
    setExpanded(false)
    setMode('drill')
    window.scrollTo({ top: 0 })
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

  const HotButton = ({ target }: { target: Question }) => (
    <button
      onClick={(e) => {
        e.stopPropagation()
        toggleHighFreq(target)
      }}
      title={target.meta.highfreq ? '取消高频标记' : '标记为高频题'}
      className={[
        'w-6 shrink-0 border text-xs transition-colors',
        target.meta.highfreq
          ? 'border-red-500 bg-red-500 font-semibold text-white'
          : 'border-gray-200 text-gray-300 hover:border-red-300 hover:text-red-400',
      ].join(' ')}
    >
      高
    </button>
  )

  if (mode === 'list') {
    return (
      <main className="mx-auto max-w-3xl px-6 py-16">
        <div className="flex items-baseline justify-between">
          <h1 className="text-2xl font-bold">{category}</h1>
          <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">
            ← 返回主页
          </Link>
        </div>
        <p className="mt-2 text-sm text-gray-500">
          共 {total} 题 · 已掌握 {masteredCount} · 高频 {highFreqCount};按 {groups.length} 个主题组织。
          点题目进入刷题界面(Esc 返回列表),点「高」标记高频(自动排到组内最前)。
        </p>
        <button
          onClick={() => openDrill(0)}
          className="mt-4 border border-gray-900 bg-gray-900 px-4 py-2 text-sm text-white hover:bg-gray-700"
        >
          从第一题开始刷 →
        </button>

        {groups.map((g) => (
          <section key={g.name} className="mt-8">
            <h2 className="mb-2 flex items-baseline gap-2 font-mono text-xs tracking-widest text-gray-400">
              {g.name}
              <span className="text-gray-300">{g.items.length} 题</span>
            </h2>
            <div className="border border-gray-200 bg-white">
              {g.items.map(({ item, index: i }, row) => (
                <div
                  key={item.file}
                  onClick={() => openDrill(i)}
                  className={[
                    'flex cursor-pointer items-center gap-3 px-4 py-2.5 hover:bg-gray-50',
                    row > 0 ? 'border-t border-gray-100' : '',
                  ].join(' ')}
                >
                  <span className="w-6 shrink-0 text-center text-xs text-green-600">
                    {item.meta.mastered ? '✓' : ''}
                  </span>
                  <span
                    className={`flex-1 truncate text-sm ${item.meta.mastered ? 'text-gray-400' : 'text-gray-900'}`}
                  >
                    {item.error ? `⚠️ ${item.file}` : (item.meta.summary ?? item.file)}
                  </span>
                  {item.sections['Note']?.trim() && (
                    <span title="有笔记" className="shrink-0 text-xs opacity-60">
                      📝
                    </span>
                  )}
                  {item.meta.company && (
                    <span className="hidden shrink-0 text-xs text-gray-400 sm:inline">{item.meta.company}</span>
                  )}
                  <span
                    className={`w-10 shrink-0 text-right text-xs ${DIFFICULTY_STYLE[item.meta.difficulty ?? ''] ?? 'text-gray-400'}`}
                  >
                    {item.meta.difficulty ?? ''}
                  </span>
                  <HotButton target={item} />
                </div>
              ))}
            </div>
          </section>
        ))}

        {toast && (
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-gray-900 px-4 py-2 text-sm text-white">
            {toast}
          </div>
        )}
      </main>
    )
  }

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
        <div className="flex items-center gap-3 text-sm text-gray-500">
          <button onClick={() => setMode('list')} className="hover:text-gray-900">
            ← 返回列表
          </button>
          <span className="text-gray-300">|</span>
          <Link href="/" className="hover:text-gray-900">
            返回主页
          </Link>
        </div>
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
              <HotButton target={q} />
            </div>

            <div className="prose mt-4 max-w-none font-semibold">
              <Markdown>{q.sections['题目'] ?? ''}</Markdown>
            </div>

            {(q.meta.tags.length > 0 || q.meta.company || kbHref) && (
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
                {kbHref && (
                  <Link
                    href={kbHref}
                    className="border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs text-blue-600 hover:border-blue-400"
                  >
                    📚 知识库:{topicHead}
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
        <span className="font-mono text-xs text-gray-400">←/→ 切题 · 空格 展开 · Esc 回列表</span>
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
