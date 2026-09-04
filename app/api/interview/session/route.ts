import { NextResponse } from 'next/server'
import { buildCorpus, loadQuestionDetail, loadToneSamples } from '@/lib/interview/corpus'
import { resolveMaterial } from '@/lib/interview/material'
import { buildSession, type QuestionRecord } from '@/lib/interview/context'
import { chat } from '@/lib/interview/llm'
import { articleBody } from '@/lib/interview/kbdrill'
import { listResumes, loadProfile } from '@/lib/interview/resume'
import { reviewPrompt } from '@/lib/interview/prompts'
import { writeSession, newSessionId, type SessionTurn } from '@/lib/interview/session'

export const dynamic = 'force-dynamic'

/**
 * 结束一场面试:生成复盘 → 连同逐轮记录一起落盘到 `interview/sessions/`。
 *
 * 复盘走 `post` 阶段(可用 `INTERVIEW_LLM_POST_*` 单独指向更强的模型),
 * 而且**思考全开** —— 一场只跑一次、你在读报告,慢一点无所谓,要的是写得准。
 * 这和现场判卷刚好相反,那边是关思考求快。
 */
export async function POST(req: Request) {
  let body: Record<string, unknown>
  try {
    body = (await req.json()) as Record<string, unknown>
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }

  // 单篇过题:上下文来源是文章而不是简历
  const kbArticle = typeof body.article === 'string' ? body.article : ''
  const resume = kbArticle
    ? { slug: '', name: kbArticle, role: '', chapters: [], chapterWeights: {}, isDefault: false, body: articleBody(kbArticle) ?? '' }
    : listResumes().find((r) => r.slug === String(body.resume ?? ''))
  if (!resume) return NextResponse.json({ error: '没有这份简历' }, { status: 400 })

  const turns = (Array.isArray(body.turns) ? body.turns : []) as SessionTurn[]
  if (!turns.length) return NextResponse.json({ error: '这场还没有任何记录' }, { status: 400 })

  const mode = body.mode === 'drill' ? 'drill' : 'interview'
  const corpus = buildCorpus()

  // 要点一律**服务端查表补**,不用前端传来的。
  // 面试档下参考答案根本不下发浏览器(防作弊的结构保证),所以前端的 `points` 恒为空 ——
  // 不补的话落盘的分母永远是 0(`要点命中:11/0`),逐轮记录里一个 ✅❌ 都没有。
  // 和「追问只给序号、正文查表还原」是同一个思路:不信前端,查表。
  const profile = loadProfile(resume.slug)
  const sessionSeed = typeof body.sessionSeed === 'number' ? body.sessionSeed : 0
  const 要点Of = new Map<string, string[]>()
  for (const id of new Set(turns.map((t) => t.questionId))) {
    const d = resolveMaterial(id, resume, profile, sessionSeed)
    if (d) 要点Of.set(id, d.要点)
  }
  const filled: SessionTurn[] = turns.map((t) => ({
    ...t,
    // 考点行服务端查出来是空数组(清单是模型现抽的,存在前端传来的 points 里)——
    // 用 ?? 会被空数组盖掉,必须判长度
    points: 要点Of.get(t.questionId)?.length ? 要点Of.get(t.questionId)! : (t.points ?? []),
  }))

  // 用整场对话当上下文,复盘才看得到每一轮的原话、判分和面试官备注
  const history: QuestionRecord[] = []
  for (const id of [...new Set(turns.map((t) => t.questionId))]) {
    // 走 resolveMaterial 而不是 loadQuestionDetail:后者读不出阶段题(自我介绍 / 项目深挖),
    // 那样复盘会看不到整场最前面的项目部分,只剩技术题。
    const d = resolveMaterial(id, resume, profile, sessionSeed)
    if (!d) continue
    const mine = turns.filter((t) => t.questionId === id)
    history.push({
      material: {
        题目: d.题目,
        要点: d.要点,
        答案: d.答案,
        追问: d.追问,
        needsReview: d.needsReview,
        examPoints: corpus.articles.find((a) => a.title === d.article)?.examPoints ?? [],
      },
      turns: mine.map((t) => ({
        ask: t.ask,
        answer: t.answer,
        verdict: JSON.stringify({ hit: t.hit, miss: t.miss, note: t.note }),
      })),
    })
  }

  // history 空 = 每一条 questionId 都查不到材料。以前这里静默留空,
  // 页面只显示「(没有生成复盘)」,看不出是模型失败还是数据对不上 —— 踩过一次:
  // 前端把 `phase:project:0` 砍成了 `ase:project:0`,整场没有复盘还不报错。
  let review =
    history.length === 0
      ? `(没能生成复盘:${turns.length} 轮记录里没有一条能查到判分材料,` +
        `questionId 依次是 ${[...new Set(turns.map((t) => t.questionId))].join('、')})`
      : ''
  if (history.length) {
    try {
      const built = buildSession({
        mode,
        toneSamples: loadToneSamples(),
        resumeName: resume.name,
        resumeBody: resume.body,
        sourceKind: kbArticle ? 'article' : 'resume',
        history: history.slice(0, -1),
        current: history[history.length - 1],
      })
      const msgs = [...built.messages, { role: 'user' as const, content: reviewPrompt() }]
      // ⚠️ `max_tokens` 是**思考与正文共用**的预算。复盘这条是故意开思考的,
      // 给 4000 时实测思考自己就烧掉 3998,正文零字、finish_reason=length ——
      // 表现是「(没有生成复盘)」。开思考的调用必须把预算按思考量放宽。
      const res = await chat('post', msgs, { maxTokens: 16000, noThink: false })
      review = res.content.trim()
    } catch (e) {
      // 复盘失败不能吃掉转录 —— 记录照样落盘,复盘位置写明原因
      review = `(复盘生成失败:${(e as Error).message})`
    }
  }

  const id = newSessionId(kbArticle ? `kb-${kbArticle}` : resume.slug)
  const file = writeSession(
    {
      id,
      startedAt: String(body.startedAt ?? new Date().toISOString()),
      resume: resume.name,
      mode,
      scope: kbArticle ? `单篇 / ${kbArticle}` : undefined,
      voice: body.voice as string | undefined,
    },
    filled,
    review,
  )
  return NextResponse.json({ id, file: file.replace(`${process.cwd()}/`, ''), review })
}
