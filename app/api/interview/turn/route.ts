import { NextResponse } from 'next/server'
import { buildCorpus, loadQuestionDetail, loadToneSamples } from '@/lib/interview/corpus'
import { buildSession, type QuestionRecord } from '@/lib/interview/context'
import { chat, extractJson, resolveTurn } from '@/lib/interview/llm'
import type { RawTurn } from '@/lib/interview/llm'
import { listResumes } from '@/lib/interview/resume'
import type { Mode } from '@/lib/interview/prompts'

export const dynamic = 'force-dynamic'

/** 前端回传的一道题:题目 id + 这道题下面的问答轮次 */
interface WireQuestion {
  questionId: string
  turns: { ask: string; answer: string; verdict?: string }[]
}

/**
 * 一轮问答:交回答 → 判卷 + 决定下一步。
 *
 * **服务端无状态**:整场进度由前端持有并回传,服务重建不打断面试。
 * 判分材料由服务端按 questionId 从磁盘读 —— 参考答案永远不下发浏览器,
 * 面试档下你打开 devtools 也偷看不到。
 */
export async function POST(req: Request) {
  let body: Record<string, unknown>
  try {
    body = (await req.json()) as Record<string, unknown>
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }

  const mode: Mode = body.mode === 'drill' ? 'drill' : 'interview'
  const resume = listResumes().find((r) => r.slug === String(body.resume ?? ''))
  if (!resume) return NextResponse.json({ error: `没有这份简历` }, { status: 400 })

  const wireHistory = Array.isArray(body.history) ? (body.history as WireQuestion[]) : []
  const wireCurrent = body.current as WireQuestion | undefined
  if (!wireCurrent?.questionId || !wireCurrent.turns?.length) {
    return NextResponse.json({ error: '缺少当前题目或回答' }, { status: 400 })
  }
  if (!wireCurrent.turns[wireCurrent.turns.length - 1]?.answer?.trim()) {
    return NextResponse.json({ error: '回答是空的' }, { status: 400 })
  }

  const corpus = buildCorpus()
  const toRecord = (w: WireQuestion): QuestionRecord | null => {
    const d = loadQuestionDetail(w.questionId)
    if (!d) return null
    const article = corpus.articles.find((a) => a.title === d.entry.article)
    return {
      material: {
        题目: d.题目,
        要点: d.要点,
        答案: d.答案,
        追问: d.追问,
        needsReview: d.entry.needsReview,
        examPoints: article?.examPoints ?? [],
      },
      turns: (w.turns ?? []).map((t) => ({
        ask: String(t.ask ?? ''),
        answer: String(t.answer ?? ''),
        verdict: t.verdict ? String(t.verdict) : undefined,
      })),
    }
  }

  const current = toRecord(wireCurrent)
  if (!current) return NextResponse.json({ error: `题目读不出来:${wireCurrent.questionId}` }, { status: 400 })
  const history = wireHistory.map(toRecord).filter((r): r is QuestionRecord => r !== null)

  const built = buildSession({
    mode,
    toneSamples: loadToneSamples(),
    resumeName: resume.name,
    resumeBody: resume.body,
    history,
    current,
  })

  let res
  try {
    res = await chat('live', built.messages, { maxTokens: mode === 'drill' ? 400 : 200 })
  } catch (e) {
    return NextResponse.json({ error: `模型调用失败:${(e as Error).message}` }, { status: 502 })
  }

  const detail = loadQuestionDetail(wireCurrent.questionId)!
  const d = resolveTurn(extractJson<RawTurn>(res.content), detail.追问, detail.要点.length)
  const article = corpus.articles.find((a) => a.title === detail.entry.article)

  return NextResponse.json({
    hit: d.hit,
    miss: d.miss,
    next: d.next,
    followUp: d.followUp,
    source: d.source,
    // 判卷 JSON 原样回传,前端存进转录 —— 下一轮它会作为 assistant 消息进上下文
    verdict: res.content.trim(),
    // 过题档一题一结算;面试档一律不给
    ...(mode === 'drill'
      ? {
          要点: detail.要点,
          答案: detail.答案,
          评价: d.comment ?? '',
          去看: article ? { 文章: article.title, 章节: article.chapter } : null,
        }
      : {}),
    debug: {
      elapsedMs: res.elapsedMs,
      promptTokens: res.promptTokens,
      cachedTokens: res.cachedTokens,
      outputTokens: res.outputTokens,
      estimated: built.estimatedTokens,
      overSoftLimit: built.overSoftLimit,
    },
  })
}
