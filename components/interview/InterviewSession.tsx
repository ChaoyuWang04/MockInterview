'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import Markdown from '@/components/Markdown'
import VoiceSettings from '@/components/interview/VoiceSettings'
import { useVoice, useVoiceConfig, collectHotwords } from '@/components/interview/useVoice'
import SttBench from '@/components/interview/SttBench'
import { PHASE_CAPS, PHASE_LABEL, PHASE_ORDER, type InterviewPhase } from '@/lib/interview/phases'

export interface ResumeOption {
  slug: string
  name: string
  role: string
  chapters: string[]
  isDefault: boolean
  hasProfile: boolean
}

type Mode = 'interview' | 'drill'

interface SttOption {
  id: string
  label: string
  ready: boolean
}

interface Question {
  id: string
  article: string
  chapter: string
  题目: string
  要点数: number
  追问数: number
  highfreq: boolean
  真题: boolean
  待校对: boolean
  poolSize: number
  phase?: InterviewPhase
  phaseLabel?: string
  projectCount?: number
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
  /** 抽到这题时的阶段。**定死一次不再改** —— 它进上下文的材料块,一变前缀缓存就断 */
  phase?: InterviewPhase
  turns: { ask: string; answer: string; verdict?: string }[]
}

/**
 * 抽题接口返回的 id → 回传服务端用的 questionId。
 *
 * **只有 `q:` 开头的题库题才剥前缀**,别的一律原样传:
 *   `q:<分类>/<文件>`   → 剥掉 `q:`
 *   `phase:intro` / `phase:project:0` → 原样
 *   `k:<文章名>#<行号>`(单篇过题的考点行) → 原样
 *
 * ⚠️ 这里**必须用白名单,不能用排除法**。写成「不是 phase: 就 slice(2)」的版本
 * 栽过两次:先把 `phase:project:0` 砍成 `ase:project:0`(整场没有复盘,不报错),
 * 后来加了考点行又把 `k:CudaGraph#9` 砍成 `CudaGraph#9`(题目读不出来)。
 * 每加一种新 id 排除法就再坏一次,白名单不会。
 */
function toQuestionId(id: string): string {
  return id.startsWith('q:') ? id.slice(2) : id
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

/** 单篇过题的范围;传了它就不选简历不选模式,固定过题档、池子只限这一篇 */
export interface KbScope {
  article: string
  chapter: string
  /** 题库题 + 考点行 */
  poolSize: number
  questionCount: number
  examPointCount: number
}

export default function InterviewSession({
  resumes,
  scope,
}: {
  resumes: ResumeOption[]
  scope?: KbScope
}) {
  const [resume, setResume] = useState(resumes.find((r) => r.isDefault)?.slug ?? resumes[0]?.slug ?? '')
  const [mode, setMode] = useState<Mode>(scope ? 'drill' : 'interview')
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
  /** finish 的最新引用 —— nextQuestion 里「单篇过完了」要调它,但 finish 定义在后面 */
  const finishRef = useRef<(() => Promise<void>) | null>(null)
  const [debug, setDebug] = useState<string>('')
  // 存 localStorage:改一次永久生效,刷新不丢
  const [voice, setVoice] = useVoiceConfig()
  const [recording, setRecording] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const { speak, stopSpeak, startRecording, stopAndTranscribe } = useVoice()
  const [sttOptions, setSttOptions] = useState<SttOption[]>([])
  const [showVoice, setShowVoice] = useState(false)
  const [finished, setFinished] = useState<{ review: string; file: string } | null>(null)
  const [ending, setEnding] = useState(false)
  const [pendingSubmit, setPendingSubmit] = useState<string | null>(null)
  const startedAt = useRef('')

  // 语音服务的可用后端列表;服务没起也不影响打字面试。
  // 顺带做一次回落:默认引擎是云端的,没配 DASHSCOPE / 没下模型时别把人卡在不可用的选项上。
  useEffect(() => {
    fetch('/api/interview/voice')
      .then((r) => r.json())
      .then((d) => {
        const stt: SttOption[] = d.stt ?? []
        setSttOptions(stt)
        setVoice((v) => {
          if (stt.find((s) => s.id === v.sttModel)?.ready) return v
          const usable = stt.find((s) => s.ready)
          return usable ? { ...v, sttModel: usable.id } : v
        })
      })
      .catch(() => setSttOptions([]))
  }, [setVoice])

  // 服务端无状态,整场进度全在这里;刷新页面会重来,但服务重建不影响。
  // 结构对齐上下文的 append-only 设计:每道题一条记录,里面是这道题下的问答轮次。
  const asked = useRef<string[]>([])
  const history = useRef<WireQuestion[]>([])
  const current = useRef<WireQuestion | null>(null)
  const followUpRound = useRef(0)

  // ── 面试档的阶段机(过题档全程停在 breadth,行为和以前一模一样)──
  // 阶段边界由代码控制:模型可以发 nextphase 提前结束,但轮次上限卡死在 PHASE_CAPS。
  const phase = useRef<InterviewPhase>('breadth')
  const projectIndex = useRef(0)
  const phaseRounds = useRef(0)
  const projectTotal = useRef(0)
  /** 本场已经深挖过的项目 —— 不重复挑 */
  const usedProjects = useRef<Set<number>>(new Set())
  /**
   * 会话种子。整场不变,随每次请求回传:
   * ① 服务端据此决定开场材料里项目清单的呈现顺序(所以每场顺序不同)
   * ② 模型没挑项目时,前端用它做可复现的随机兜底
   * **必须整场一致** —— 变了开场材料就变了,前缀缓存会断。
   */
  const sessionSeed = useRef(0)
  const [phaseLabel, setPhaseLabel] = useState('')

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

  /**
   * 挑下一个要深挖的项目。**优先用模型选的**(它刚听完你的自我介绍或上一个项目),
   * 模型没选 / 选越界 / 选了已问过的,就用会话种子随机挑一个没问过的。
   * 返回 null 表示没有可挑的了。
   */
  const pickProject = useCallback((suggested?: number): number | null => {
    const left = Array.from({ length: projectTotal.current }, (_, i) => i).filter(
      (i) => !usedProjects.current.has(i),
    )
    if (left.length === 0) return null
    if (suggested !== undefined && left.includes(suggested)) return suggested
    // 兜底随机:模型不配合不能让面试卡住,也不能永远退化成「总是第 0 个」
    return left[(sessionSeed.current + usedProjects.current.size) % left.length]
  }, [])

  /**
   * 推进阶段。
   *
   * **不再按 0,1,2 把每个项目挨个问一遍** —— 真面试官会挑一两个挖透,剩下的不碰。
   * 挑哪个由模型说了算(`proj` 字段),挑几个由模型说了算(它随时可以转广度或收尾)。
   * 代码只保证:不重复挑、越界有兜底、轮次有上限。
   */
  const advancePhase = useCallback(
    (suggestedProject?: number) => {
      phaseRounds.current = 0
      if (phase.current === 'project') {
        // 深挖完 → 进这个项目的技术延伸
        phase.current = 'tech'
        return
      }
      if (phase.current === 'tech' || phase.current === 'intro') {
        const next = pickProject(suggestedProject)
        if (next !== null) {
          usedProjects.current.add(next)
          projectIndex.current = next
          phase.current = 'project'
          return
        }
        phase.current = 'breadth'
        return
      }
      const i = PHASE_ORDER.indexOf(phase.current)
      phase.current = PHASE_ORDER[Math.min(i + 1, PHASE_ORDER.length - 1)]
    },
    [pickProject],
  )

  const nextQuestion = useCallback(async () => {
    setBusy(true)
    setError('')
    setSettlement(null)
    try {
      let q: Question | null = null
      // 阶段题可能取不到(没有画像、项目问完了、关联不到题)—— 顺次往下走,最多走到 breadth。
      // 上限给足:project↔tech 会来回切,每个项目各占一次
      for (let attempt = 0; attempt < PHASE_ORDER.length + 8 && !q; attempt++) {
        try {
          q = (await post('/api/interview/next', {
            resume,
            asked: asked.current,
            lastArticle: question?.article,
            phase: phase.current,
            projectIndex: projectIndex.current,
            sessionSeed: sessionSeed.current,
            article: scope?.article,
          })) as Question
        } catch (e) {
          const msg = (e as Error).message
          // 阶段取不到题是**正常情况**,不是错误:没有画像、项目问完了、
          // 或者这个项目关联的文章在题库里一道题都没有。顺次往下走就是了。
          if (msg.includes('ARTICLE_DONE')) {
            // 这一篇全过完了 —— 直接收尾并生成复盘,不再抽题
            await finishRef.current?.()
            return
          }
          if (msg.includes('NO_MORE_PROJECTS')) {
            phase.current = 'breadth'
            phaseRounds.current = 0
          } else if (msg.includes('NO_RELATED_QUESTIONS')) {
            advancePhase() // 这个项目接不上技术题 —— 换项目或转广度
          } else {
            throw e
          }
        }
      }
      if (!q) throw new Error('抽不到题:阶段都走完了,候选池也是空的')

      asked.current = [...asked.current, q.id]
      if (q.projectCount) projectTotal.current = q.projectCount
      // 上一题(如果有)整条归档进 history,当前题另起一条 —— 只追加,不回改
      if (current.current) history.current = [...history.current, current.current]
      // 阶段题的 id 是 `phase:xxx`,不能像题库 id 那样剥掉前两个字符
      current.current = { questionId: toQuestionId(q.id), phase: q.phase, turns: [] }
      followUpRound.current = 0
      phaseRounds.current = 0
      setPhaseLabel(q.phaseLabel ?? '')
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
    return collectHotwords(terms)
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
        sessionSeed: sessionSeed.current,
        article: scope?.article,
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
    // 面试档从自我介绍开场;过题档没有阶段,直接进正题
    phase.current = mode === 'interview' && !scope ? 'intro' : 'breadth'
    projectIndex.current = 0
    phaseRounds.current = 0
    usedProjects.current = new Set()
    // 每场一个新种子 —— 项目清单的呈现顺序因此每场都不同,
    // 即使模型的偏好稳定,落到的项目也会变。这是「同一份简历多跑几场有价值」的前提。
    sessionSeed.current = Math.floor(Math.random() * 1e9)
    await nextQuestion()
  }, [nextQuestion, mode])

  const submit = useCallback(async () => {
    if (!question || !answer.trim() || busy) return
    setBusy(true)
    setError('')
    try {
      // 把这一轮追加进当前题,再整体回传 —— 服务端据此复原整场 append-only 上下文。
      // **判卷成功之前不写回 cur.turns**:失败时不留半条记录,否则你再点一次
      // 「答完了」会把同一问同一答塞进上下文两遍。
      const cur = current.current!
      const pending = { ask, answer }
      const r = await post('/api/interview/turn', {
        resume,
        mode,
        history: history.current,
        current: { ...cur, turns: [...cur.turns, pending] },
        sessionSeed: sessionSeed.current,
        article: scope?.article,
      })
      // 判卷结果存回这一轮:下一次它会作为 assistant 消息进上下文,面试官因此记得判过什么
      cur.turns = [...cur.turns, { ...pending, verdict: r.verdict }]
      setTurns((t) => [
        ...t,
        {
          ask,
          answer,
          article: question.article,
          questionId: toQuestionId(question.id),
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
      // 阶段内追问归模型(`followup` / `nextq`),**阶段边界归代码**:
      // 模型发 `nextphase` 可以提前结束一个阶段,但轮次到了上限一律强制推进 ——
      // 不然一场面试可能在自我介绍上耗掉二十轮。
      followUpRound.current += 1
      phaseRounds.current += 1
      const capped = phaseRounds.current >= PHASE_CAPS[phase.current]
      // 开场与项目深挖**一个阶段只有一道主问**,没有「下一题」可换。
      // 模型发 nextq 时若不当成 nextphase,会原地反复拿到同一道题(实测踩到)。
      const oneQuestionPhase = phase.current === 'intro' || phase.current === 'project'

      if (r.next === 'end') {
        await finish()
      } else if (r.next === 'nextphase' || capped || (oneQuestionPhase && r.next === 'nextq')) {
        advancePhase(typeof r.project === 'number' ? r.project : undefined)
        await nextQuestion()
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
  }, [question, answer, busy, post, resume, mode, ask, nextQuestion, finish, advancePhase])

  useEffect(() => {
    submitRef.current = submit
  }, [submit])

  useEffect(() => {
    finishRef.current = finish
  }, [finish])

  /**
   * 「这题我熟,跳过」。
   *
   * ⚠️ 面试档下必须**推进阶段**,不能只调 nextQuestion:开场和项目深挖
   * 一个阶段只有一道主问,原地再抽一次拿到的还是同一道题(这个按钮在面试档下
   * 因此一度是失灵的)。技术延伸和广度补充有题池,跳过 = 换一道,阶段不变。
   */
  const skip = useCallback(async () => {
    if (!question) return
    setTurns((t) => [
      ...t,
      {
        ask,
        answer: '(跳过)',
        article: question.article,
        questionId: toQuestionId(question.id),
        hit: [],
        miss: [],
        points: [],
        skipped: true,
      },
    ])
    // 跳过的这一轮也要进上下文 —— 否则面试官不知道你跳过了,可能再问一遍类似的
    const cur = current.current
    if (cur) cur.turns = [...cur.turns, { ask, answer: '(跳过)' }]
    if (mode === 'interview' && (phase.current === 'intro' || phase.current === 'project')) {
      advancePhase()
    }
    await nextQuestion()
  }, [question, ask, nextQuestion, mode, advancePhase])

  // ---------- 开场:选简历 + 选档 ----------
  if (!started) {
    return (
      <div className="space-y-10">
        {scope ? (
          <section className="border border-gray-300 bg-white p-5">
            <h2 className="font-semibold">单篇过题 · {scope.article}</h2>
            <p className="mt-2 text-sm leading-relaxed text-gray-600">
              只考这一篇。共 <strong>{scope.poolSize}</strong> 问
              —— 题库题 {scope.questionCount} 道 · 考点 {scope.examPointCount} 条,
              随机过一遍,过完自动收尾并生成复盘。
            </p>
            <p className="mt-2 font-mono text-[11px] leading-relaxed text-gray-400">
              判分口径:题库题对着人写的 `## 要点` 打勾;考点行没有现成清单,
              由模型从本文原文现抽 3–5 条再对照 —— 清单会连同原文一起给你看。
            </p>
          </section>
        ) : (
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
        )}

        <section className={scope ? 'hidden' : undefined}>
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

        <div className="space-y-3">
          <button
            onClick={start}
            // 单篇过题没有简历,不能拿 resume 当能否开始的条件
            disabled={(!resume && !scope) || busy}
            className="border border-gray-800 bg-gray-900 px-6 py-3 text-sm text-white transition-colors hover:bg-gray-700 disabled:opacity-40"
          >
            {busy ? '准备中…' : '开始'}
          </button>
          {error && <p className="text-sm text-red-600">{error}</p>}
        </div>

        {/* 语音设置默认收起 —— 设置存了 localStorage,调一次就长期有效,不该每次挡在「开始」前面。
            展开一次把面板和对照台一起给出来,不用再点第二层。 */}
        <section className="border-t border-gray-200 pt-6">
          <div className="flex items-baseline justify-between">
            <h2 className="text-sm font-semibold text-gray-500">语音设置</h2>
            <button
              onClick={() => setShowVoice((v) => !v)}
              className="text-xs text-gray-400 hover:text-gray-700"
            >
              {showVoice ? '收起' : '展开(音色 / 语速 / STT 对照台)→'}
            </button>
          </div>
          {showVoice ? (
            <div className="mt-3 space-y-3">
              <VoiceSettings config={voice} onChange={setVoice} />
              <SttBench options={sttOptions} />
            </div>
          ) : (
            <p className="mt-1 font-mono text-[11px] text-gray-400">
              语音模式 {voice.enabled ? '开' : '关'} · 转写{' '}
              {sttOptions.find((s) => s.id === voice.sttModel)?.label ?? voice.sttModel} · 语速{' '}
              {voice.speed.toFixed(2)}×
            </p>
          )}
        </section>
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
            {turns.length} 轮 · 已存{' '}
            <a
              href={`/interview/sessions/${finished.file.replace(/^.*\//, '').replace(/\.md$/, '')}`}
              className="underline hover:text-gray-700"
            >
              {finished.file}
            </a>
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
          {MODE_INFO[mode].name}
          {phaseLabel && (
            <span className="ml-2 bg-gray-900 px-1.5 py-0.5 text-white">{phaseLabel}</span>
          )}
          <span className="ml-2">
            第 {turns.length + 1} 轮 · 已问 {asked.current.length} 题
            {question && ` · 池 ${question.poolSize}`}
          </span>
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
          {question.真题 && <span className="bg-red-50 px-2 py-0.5 text-red-700">真题</span>}
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
