import { NextResponse } from 'next/server'
import { buildCorpus, loadQuestionDetail, loadToneSamples } from '@/lib/interview/corpus'
import { buildSession, type QuestionRecord } from '@/lib/interview/context'
import { chat } from '@/lib/interview/llm'
import { listResumes } from '@/lib/interview/resume'
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

  const resume = listResumes().find((r) => r.slug === String(body.resume ?? ''))
  if (!resume) return NextResponse.json({ error: '没有这份简历' }, { status: 400 })

  const turns = (Array.isArray(body.turns) ? body.turns : []) as SessionTurn[]
  if (!turns.length) return NextResponse.json({ error: '这场还没有任何记录' }, { status: 400 })

  const mode = body.mode === 'drill' ? 'drill' : 'interview'
  const corpus = buildCorpus()

  // 用整场对话当上下文,复盘才看得到每一轮的原话、判分和面试官备注
  const history: QuestionRecord[] = []
  for (const id of [...new Set(turns.map((t) => t.questionId))]) {
    const d = loadQuestionDetail(id)
    if (!d) continue
    const mine = turns.filter((t) => t.questionId === id)
    history.push({
      material: {
        题目: d.题目,
        要点: d.要点,
        答案: d.答案,
        追问: d.追问,
        needsReview: d.entry.needsReview,
        examPoints: corpus.articles.find((a) => a.title === d.entry.article)?.examPoints ?? [],
      },
      turns: mine.map((t) => ({
        ask: t.ask,
        answer: t.answer,
        verdict: JSON.stringify({ hit: t.hit, miss: t.miss, note: t.note }),
      })),
    })
  }

  let review = ''
  if (history.length) {
    try {
      const built = buildSession({
        mode,
        toneSamples: loadToneSamples(),
        resumeName: resume.name,
        resumeBody: resume.body,
        history: history.slice(0, -1),
        current: history[history.length - 1],
      })
      const msgs = [...built.messages, { role: 'user' as const, content: reviewPrompt() }]
      const res = await chat('post', msgs, { maxTokens: 4000, noThink: false })
      review = res.content.trim()
    } catch (e) {
      // 复盘失败不能吃掉转录 —— 记录照样落盘,复盘位置写明原因
      review = `(复盘生成失败:${(e as Error).message})`
    }
  }

  const id = newSessionId(resume.slug)
  const file = writeSession(
    {
      id,
      startedAt: String(body.startedAt ?? new Date().toISOString()),
      resume: resume.name,
      mode,
      voice: body.voice as string | undefined,
    },
    turns,
    review,
  )
  return NextResponse.json({ id, file: file.replace(`${process.cwd()}/`, ''), review })
}
