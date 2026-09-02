'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import Markdown from '@/components/Markdown'
import VoiceSettings from '@/components/interview/VoiceSettings'
import { useVoice, expandHotwords, DEFAULT_VOICE_CONFIG } from '@/components/interview/useVoice'
import type { VoiceConfig } from '@/components/interview/useVoice'
import SttBench from '@/components/interview/SttBench'

export interface ResumeOption {
  slug: string
  name: string
  role: string
  chapters: string[]
  isDefault: boolean
  hasProfile: boolean
}

type Mode = 'interview' | 'drill'

interface Question {
  id: string
  article: string
  chapter: string
  题目: string
  要点数: number
  追问数: number
  highfreq: boolean
  面经: boolean
  待校对: boolean
  poolSize: number
}

interface Settlement {
  hit: number[]
  miss: number[]
  要点?: string[]
  答案?: string
  评价?: string
  去看?: { 文章: string; 章节: string } | null
}

interface Turn {
  ask: string
  answer: string
  article: string
  questionId: string
  hit: number[]
  miss: number[]
  points: string[]
  note?: string
  skipped?: boolean
}

/** 回传给服务端的一道题:题目 id + 这道题下面的问答轮次(与上下文的 append-only 结构对齐) */
interface WireQuestion {
  questionId: string
  turns: { ask: string; answer: string; verdict?: string }[]
}

const MODE_INFO: Record<Mode, { name: string; desc: string }> = {
  interview: {
    name: '面试档',
    desc: '连续追问,全程不给答案。像真面试一样走完,结束时才复盘。',
  },
  drill: {
    name: '过题档',
    desc: '一题一结算,答完立刻看命中了哪些要点、标准答案是什么。可以无限过下去。',
  },
}

export default function InterviewSession({ resumes }: { resumes: ResumeOption[] }) {
  const [resume, setResume] = useState(resumes.find((r) => r.isDefault)?.slug ?? resumes[0]?.slug ?? '')
  const [mode, setMode] = useState<Mode>('interview')
  const [started, setStarted] = useState(false)

  const [question, setQuestion] = useState<Question | null>(null)
  const [ask, setAsk] = useState('')
  const [answer, setAnswer] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [settlement, setSettlement] = useState<Settlement | null>(null)
  const [turns, setTurns] = useState<Turn[]>([])
  // finish() 里要拿最新的 turns,而 setState 是异步的 —— 用 ref 同步一份
  const turnsRef = useRef<Turn[]>([])
  /** submit 的最新引用 —— 自动提交的 effect 要用,但 submit 定义在它后面 */
  const submitRef = useRef<(() => Promise<void>) | null>(null)
  const [debug, setDebug] = useState<string>('')
  const [voice, setVoice] = useState<VoiceConfig>(DEFAULT_VOICE_CONFIG)
  const [recording, setRecording] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const { speak, stopSpeak, startRecording, stopAndTranscribe } = useVoice()
  const [sttOptions, setSttOptions] = useState<{ id: string; label: string; ready: boolean }[]>([])
  const [showBench, setShowBench] = useState(false)
  const [finished, setFinished] = useState<{ review: string; file: string } | null>(null)
  const [ending, setEnding] = useState(false)
  const [pendingSubmit, setPendingSubmit] = useState<string | null>(null)
  const startedAt = useRef('')

  // 语音服务的可用后端列表;服务没起也不影响打字面试
  useEffect(() => {
    fetch('/api/interview/voice')
      .then((r) => r.json())
      .then((d) => setSttOptions(d.stt ?? []))
      .catch(() => setSttOptions([]))
  }, [])

  // 服务端无状态,整场进度全在这里;刷新页面会重来,但服务重建不影响。
  // 结构对齐上下文的 append-only 设计:每道题一条记录,里面是这道题下的问答轮次。
  const asked = useRef<string[]>([])
  const history = useRef<WireQuestion[]>([])
  const current = useRef<WireQuestion | null>(null)
  const followUpRound = useRef(0)

  useEffect(() => {
    turnsRef.current = turns
  }, [turns])

  const post = useCallback(async (url: string, body: unknown) => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
    return data
  }, [])

  const nextQuestion = useCallback(async () => {
    setBusy(true)
    setError('')
    setSettlement(null)
    try {
      const q: Question = await post('/api/interview/next', {
        resume,
        asked: asked.current,
        lastArticle: question?.article,
      })
      asked.current = [...asked.current, q.id]
      // 上一题(如果有)整条归档进 history,当前题另起一条 —— 只追加,不回改
      if (current.current) history.current = [...history.current, current.current]
      current.current = { questionId: q.id.slice(2), turns: [] }
      followUpRound.current = 0
      setQuestion(q)
      setAsk(q.题目)
      setAnswer('')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [post, resume, question?.article])

  // 语音模式下,题目和追问一出现就念出来。失败只提示,不打断面试。
  useEffect(() => {
    if (!started || !voice.enabled || !ask) return
    let cancelled = false
    setSpeaking(true)
    speak(ask, voice)
      .catch((e: Error) => !cancelled && setError(`朗读失败:${e.message}`))
      .finally(() => !cancelled && setSpeaking(false))
    return () => {
      cancelled = true
      stopSpeak()
    }
  }, [ask, started, voice, speak, stopSpeak])

  /** 转写用的热词:这道题的文章名 + 题干里出现的英文术语。
      系统知道这道题会出现什么,通用听写工具不知道 —— 这是我们的优势。 */
  const hotwords = useCallback(() => {
    const terms = new Set<string>()
    if (question?.article) terms.add(question.article)
    for (const m of ask.match(/[A-Za-z][A-Za-z0-9_.-]{1,24}/g) ?? []) terms.add(m)
    return expandHotwords(terms)
  }, [question?.article, ask])

  /** 录音开关。停止后自动转写,转写完自动提交 —— 全程不用点第二下。 */
  const toggleRecord = useCallback(async () => {
    setError('')
    if (!recording) {
      try {
        stopSpeak() // 别把面试官的声音录进去
        await startRecording()
        setRecording(true)
      } catch (e) {
        setError(`拿不到麦克风:${(e as Error).message}`)
      }
      return
    }
    setRecording(false)
    setBusy(true)
    try {
      const { text, elapsedMs } = await stopAndTranscribe(voice, hotwords())
      setDebug(`转写 ${elapsedMs}ms · ${voice.sttModel}`)
      if (text.trim()) {
        setAnswer(text)
        setPendingSubmit(text) // 交给下面的 effect 提交,避开闭包里的旧 answer
      }
    } catch (e) {
      setError(`转写失败:${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }, [recording, voice, hotwords, startRecording, stopAndTranscribe, stopSpeak])

  /** 收尾:把整场记录发去生成复盘并落盘。模型判 `end` 或你点「结束」都走这里。 */
  const finish = useCallback(async () => {
    stopSpeak()
    setEnding(true)
    setError('')
    try {
      const r = await post('/api/interview/session', {
        resume,
        mode,
        startedAt: startedAt.current,
        voice: voice.instruct,
        turns: turnsRef.current,
      })
      setFinished({ review: r.review ?? '', file: r.file ?? '' })
    } catch (e) {
      setError(`复盘生成失败:${(e as Error).message}`)
    } finally {
      setEnding(false)
    }
  }, [post, resume, mode, voice.instruct, stopSpeak])

  // 转写完成后自动提交。走 effect 而不是在 toggleRecord 里直接调,
  // 是为了让 submit 读到刚 setAnswer 的新值,而不是闭包里的旧值。
  useEffect(() => {
    if (pendingSubmit === null) return
    setPendingSubmit(null)
    void submitRef.current?.()
  }, [pendingSubmit])

  // 空格键 = 开始/停止录音。输入框里打字时不拦截。
  useEffect(() => {
    if (!started || !voice.enabled || finished) return
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return
      const el = e.target as HTMLElement
      if (el?.tagName === 'TEXTAREA' || el?.tagName === 'INPUT') return
      e.preventDefault()
      void toggleRecord()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [started, voice.enabled, finished, toggleRecord])

  const start = useCallback(async () => {
    setStarted(true)
    setFinished(null)
    setTurns([])
    turnsRef.current = []
    asked.current = []
    history.current = []
    current.current = null
    startedAt.current = new Date().toLocaleString('zh-CN')
    await nextQuestion()
  }, [nextQuestion])

  const submit = useCallback(async () => {
    if (!question || !answer.trim() || busy) return
    setBusy(true)
    setError('')
    try {
      // 把这一轮追加进当前题,再整体回传 —— 服务端据此复原整场 append-only 上下文
      const cur = current.current!
      cur.turns = [...cur.turns, { ask, answer }]
      const r = await post('/api/interview/turn', {
        resume,
        mode,
        history: history.current,
        current: cur,
      })
      // 判卷结果存回这一轮:下一次它会作为 assistant 消息进上下文,面试官因此记得判过什么
      cur.turns[cur.turns.length - 1].verdict = r.verdict
      setTurns((t) => [
        ...t,
        {
          ask,
          answer,
          article: question.article,
          questionId: question.id.slice(2),
          hit: r.hit,
          miss: r.miss,
          points: r.要点 ?? [],
          note: r.note,
        },
      ])
      const cachePct = r.debug.promptTokens
        ? Math.round((r.debug.cachedTokens / r.debug.promptTokens) * 100)
        : 0
      setDebug(
        `${r.debug.elapsedMs}ms · 输入 ${r.debug.promptTokens} tok(缓存命中 ${cachePct}%)· 输出 ${r.debug.outputTokens} tok`,
      )

      if (mode === 'drill') {
        setSettlement({ hit: r.hit, miss: r.miss, 要点: r.要点, 答案: r.答案, 评价: r.评价, 去看: r.去看 })
        return
      }
      // 面试档:不给反馈,直接往下走。
      // **追问没有轮次上限** —— 追到什么时候、什么时候换题、整场什么时候收,
      // 全由模型的 `next` 决定(followup / nextq / end)。我们只做兜底。
      followUpRound.current += 1
      if (r.next === 'end') {
        await finish()
      } else if (r.next === 'followup' && r.followUp) {
        setAsk(r.followUp)
        setAnswer('')
      } else {
        await nextQuestion()
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [question, answer, busy, post, resume, mode, ask, nextQuestion, finish])

  useEffect(() => {
    submitRef.current = submit
  }, [submit])

  const skip = useCallback(async () => {
    if (!question) return
    setTurns((t) => [
      ...t,
      {
        ask,
        answer: '(跳过)',
        article: question.article,
        questionId: question.id.slice(2),
        hit: [],
        miss: [],
        points: [],
        skipped: true,
      },
    ])
    await nextQuestion()
  }, [question, ask, nextQuestion])

  // ---------- 开场:选简历 + 选档 ----------
  if (!started) {
    return (
      <div className="space-y-10">
        <section>
          <h2 className="mb-3 text-sm font-semibold text-gray-500">选简历</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {resumes.map((r) => (
              <button
                key={r.slug}
                onClick={() => setResume(r.slug)}
                className={`border p-4 text-left transition-colors ${
                  resume === r.slug ? 'border-gray-800 bg-white' : 'border-gray-200 bg-white hover:border-gray-400'
                }`}
              >
                <div className="flex items-baseline justify-between">
                  <span className="font-semibold">{r.name}</span>
                  {!r.hasProfile && (
                    <span className="font-mono text-[11px] text-amber-600">无画像</span>
                  )}
                </div>
                <p className="mt-1 text-xs text-gray-500">{r.role}</p>
                <p className="mt-2 font-mono text-[11px] text-gray-400">
                  在册 {r.chapters.map((c) => c.replace(/^\d+-/, '')).join(' · ')}
                </p>
              </button>
            ))}
          </div>
        </section>

        <section>
          <h2 className="mb-3 text-sm font-semibold text-gray-500">选模式</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {(Object.keys(MODE_INFO) as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`border p-4 text-left transition-colors ${
                  mode === m ? 'border-gray-800 bg-white' : 'border-gray-200 bg-white hover:border-gray-400'
                }`}
              >
                <div className="font-semibold">{MODE_INFO[m].name}</div>
                <p className="mt-1 text-xs leading-relaxed text-gray-500">{MODE_INFO[m].desc}</p>
              </button>
            ))}
          </div>
        </section>

        <section>
          <div className="mb-3 flex items-baseline justify-between">
            <h2 className="text-sm font-semibold text-gray-500">语音</h2>
            <button
              onClick={() => setShowBench((v) => !v)}
              className="text-xs text-gray-400 hover:text-gray-700"
            >
              {showBench ? '收起 STT 对照台' : 'STT 对照台 →'}
            </button>
          </div>
          <div className="space-y-3">
            <VoiceSettings config={voice} onChange={setVoice} />
            {showBench && <SttBench options={sttOptions} />}
          </div>
        </section>

        <button
          onClick={start}
          disabled={!resume || busy}
          className="border border-gray-800 bg-gray-900 px-6 py-3 text-sm text-white transition-colors hover:bg-gray-700 disabled:opacity-40"
        >
          {busy ? '准备中…' : '开始'}
        </button>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    )
  }

  // ---------- 面试结束:复盘 ----------
  if (finished) {
    return (
      <div className="space-y-5">
        <div className="flex items-baseline justify-between border-b border-gray-200 pb-3">
          <h2 className="text-lg font-semibold">复盘</h2>
          <span className="font-mono text-xs text-gray-400">
            {turns.length} 轮 · 已存 {finished.file}
          </span>
        </div>
        <div className="prose prose-sm max-w-none">
          <Markdown>{finished.review || '(没有生成复盘)'}</Markdown>
        </div>
        <div className="flex gap-3 border-t border-gray-200 pt-4">
          <button
            onClick={() => { setFinished(null); setStarted(false) }}
            className="border border-gray-800 bg-gray-900 px-5 py-2 text-sm text-white transition-colors hover:bg-gray-700"
          >
            再来一场
          </button>
        </div>
      </div>
    )
  }

  // ---------- 面试进行中 ----------
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between border-b border-gray-200 pb-3">
        <div className="font-mono text-xs text-gray-500">
          {MODE_INFO[mode].name} · 第 {turns.length + 1} 轮 · 已问 {asked.current.length} 题
          {question && ` · 池 ${question.poolSize}`}
        </div>
        <button
          onClick={finish}
          disabled={ending || turns.length === 0}
          className="text-xs text-gray-500 hover:text-gray-900 disabled:opacity-40"
          title="结束面试并生成复盘"
        >
          {ending ? '生成复盘中…' : '结束并复盘'}
        </button>
      </div>

      {question && (
        <div className="flex flex-wrap gap-2 font-mono text-[11px]">
          <span className="bg-gray-100 px-2 py-0.5 text-gray-600">{question.article}</span>
          {question.面经 && <span className="bg-red-50 px-2 py-0.5 text-red-700">面经真题</span>}
          {question.highfreq && <span className="bg-amber-50 px-2 py-0.5 text-amber-700">高频</span>}
          {question.待校对 && (
            <span className="bg-amber-50 px-2 py-0.5 text-amber-700" title="参考答案由 AI 代写,尚未人工核对">
              答案待校对
            </span>
          )}
          <span className="text-gray-400">
            要点 {question.要点数} · 追问池 {question.追问数}
          </span>
        </div>
      )}

      <div className="border-l-2 border-gray-800 bg-white py-1 pl-5">
        <div className="prose prose-sm max-w-none">
          <Markdown>{ask}</Markdown>
        </div>
      </div>

      {settlement ? (
        <Settle s={settlement} onNext={nextQuestion} busy={busy} />
      ) : (
        <>
          <textarea
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit()
            }}
            disabled={busy}
            rows={8}
            placeholder={voice.enabled ? '点「说话作答」用嘴答,或直接打字。⌘/Ctrl + Enter 提交。' : '口头怎么答就怎么打。⌘/Ctrl + Enter 提交。'}
            className="w-full border border-gray-300 p-3 font-sans text-sm leading-relaxed focus:border-gray-600 focus:outline-none disabled:bg-gray-50"
          />
          <div className="flex flex-wrap items-center gap-3">
            {voice.enabled && (
              <>
                <button
                  onClick={toggleRecord}
                  disabled={busy && !recording}
                  className={`border px-4 py-2 text-sm transition-colors disabled:opacity-40 ${
                    recording
                      ? 'border-red-600 bg-red-600 text-white hover:bg-red-500'
                      : 'border-gray-300 text-gray-700 hover:border-gray-500'
                  }`}
                >
                  {recording ? '⏹ 停止(空格)' : '🎙 说话作答(空格)'}
                </button>
                <button
                  onClick={() => void speak(ask, voice)}
                  disabled={speaking}
                  className="border border-gray-300 px-3 py-2 text-xs text-gray-600 transition-colors hover:border-gray-500 disabled:opacity-40"
                  title="再念一遍题目"
                >
                  {speaking ? '朗读中…' : '🔁 重念'}
                </button>
              </>
            )}
            <button
              onClick={submit}
              disabled={busy || !answer.trim()}
              className="border border-gray-800 bg-gray-900 px-5 py-2 text-sm text-white transition-colors hover:bg-gray-700 disabled:opacity-40"
            >
              {busy ? '判卷中…' : '答完了'}
            </button>
            <button
              onClick={skip}
              disabled={busy}
              className="border border-gray-300 px-4 py-2 text-sm text-gray-600 transition-colors hover:border-gray-500 disabled:opacity-40"
            >
              这题我熟,跳过
            </button>
            {debug && <span className="font-mono text-[11px] text-gray-400">{debug}</span>}
          </div>
        </>
      )}

      {error && <p className="border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p>}

      {turns.length > 0 && (
        <details className="border-t border-gray-200 pt-4">
          <summary className="cursor-pointer text-xs text-gray-500">本场记录 · {turns.length} 轮</summary>
          <ol className="mt-3 space-y-2">
            {turns.map((t, i) => (
              <li key={i} className="border-l border-gray-200 pl-3 text-xs">
                <div className="text-gray-700">{t.ask}</div>
                <div className="mt-0.5 font-mono text-[11px] text-gray-400">
                  {t.skipped ? '跳过' : `命中 ${t.hit.length} · 漏 ${t.miss.length}`} · {t.article}
                </div>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  )
}

function Settle({ s, onNext, busy }: { s: Settlement; onNext: () => void; busy: boolean }) {
  const total = (s.要点 ?? []).length
  return (
    <div className="space-y-4 border border-gray-300 bg-gray-50 p-5">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-2xl font-bold text-green-700">
          {s.hit.length}
          <span className="text-gray-300"> / </span>
          {total}
        </span>
        <span className="text-xs text-gray-500">要点命中</span>
      </div>

      {s.评价 && <p className="text-sm leading-relaxed text-gray-700">{s.评价}</p>}

      {total > 0 && (
        <ul className="space-y-1.5">
          {s.要点!.map((p, i) => (
            <li key={i} className="flex gap-2 text-sm">
              <span className={s.hit.includes(i) ? 'text-green-600' : 'text-red-500'}>
                {s.hit.includes(i) ? '✓' : '✗'}
              </span>
              <span className={s.hit.includes(i) ? 'text-gray-500' : 'text-gray-900'}>{p}</span>
            </li>
          ))}
        </ul>
      )}

      {s.答案 && (
        <details>
          <summary className="cursor-pointer text-xs text-gray-500">参考答案</summary>
          <div className="prose prose-sm mt-2 max-w-none">
            <Markdown>{s.答案}</Markdown>
          </div>
        </details>
      )}

      {s.去看 && (
        <p className="font-mono text-[11px] text-gray-400">
          去看:{s.去看.章节.replace(/^\d+-/, '')} / {s.去看.文章}
        </p>
      )}

      <button
        onClick={onNext}
        disabled={busy}
        className="border border-gray-800 bg-gray-900 px-5 py-2 text-sm text-white transition-colors hover:bg-gray-700 disabled:opacity-40"
      >
        {busy ? '抽题中…' : '下一题'}
      </button>
    </div>
  )
}
