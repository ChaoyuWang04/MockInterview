import { NextResponse } from 'next/server'
import { buildCorpus, loadToneSamples } from '@/lib/interview/corpus'
import { buildSession, type QuestionRecord } from '@/lib/interview/context'
import { chatJson, resolveTurn } from '@/lib/interview/llm'
import type { RawTurn } from '@/lib/interview/llm'
import { resolveMaterial } from '@/lib/interview/material'
import { PHASE_LABEL, type InterviewPhase } from '@/lib/interview/phases'
import { articleBody } from '@/lib/interview/kbdrill'
import { listResumes, loadProfile } from '@/lib/interview/resume'
import type { Mode } from '@/lib/interview/prompts'

export const dynamic = 'force-dynamic'

/** 前端回传的一道题:题目 id + 这道题下面的问答轮次 */
interface WireQuestion {
  questionId: string
  /** 抽到这道题时所处的阶段,前端定死一次后原样回传(见 toRecord 里的红线) */
  phase?: InterviewPhase
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
  // 单篇过题:上下文的第二条消息换成文章全文,不是简历。**结构一个字不用改**
  const kbArticle = typeof body.article === 'string' ? body.article : ''
  const resume = kbArticle
    ? { slug: '', name: kbArticle, role: '', chapters: [], chapterWeights: {}, isDefault: false, body: articleBody(kbArticle) ?? '' }
    : listResumes().find((r) => r.slug === String(body.resume ?? ''))
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
  const profile = loadProfile(resume.slug)
  // ⚠️ 会话种子必须和 /next 那次用的是同一个,否则开场材料里的项目清单顺序会变 ——
  // 那就等于「已写下的消息内容发生了变化」,前缀缓存从那条起全废。前端整场只生成一次。
  const sessionSeed = typeof body.sessionSeed === 'number' ? body.sessionSeed : 0

  const toRecord = (w: WireQuestion): QuestionRecord | null => {
    const d = resolveMaterial(w.questionId, resume, profile, sessionSeed)
    if (!d) return null
    const article = corpus.articles.find((a) => a.title === d.article)
    return {
      material: {
        题目: d.题目,
        要点: d.要点,
        答案: d.答案,
        追问: d.追问,
        needsReview: d.needsReview,
        examPoints: article?.examPoints ?? [],
        extra: d.extra,
        // ⚠️ 阶段名**跟着这道题走**,由前端在抽到题时定死一次并原样回传。
        // 不能在这里按「当前阶段」现算 —— 那样一道题变成历史之后材料块会变,
        // 前缀缓存从那条起全废(不报错、不变慢,只是悄悄贵 30 倍)。
        phase: w.phase ? PHASE_LABEL[w.phase] : undefined,
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
    sourceKind: kbArticle ? 'article' : 'resume',
    history,
    current,
  })

  const detail = resolveMaterial(wireCurrent.questionId, resume, profile, sessionSeed)!

  // 调用 + 解析放在同一个 try 里。**解析失败也必须回 JSON** ——
  // 以前 extractJson 直接抛在 try 外面,路由崩掉返回空 body,
  // 浏览器只看到 “Unexpected end of JSON input”,完全指不出问题在哪。
  let res, d
  try {
    const out = await chatJson<RawTurn>('live', built.messages, {
      // 过题档要多吐一份「评价」;考点题还要再多吐一份现抽的要点清单。
      // 给 400 时实测「要点」经常整个丢失(返回 0/0,判分等于没做)—— 预算不够时
      // 模型是**悄悄少输出字段**,不是报错。要点题单独放宽。
      maxTokens: mode === 'drill' ? (detail.要点.length ? 600 : 1200) : 200,
    })
    res = out.res
    // 把这道题下面已经问过的话传进去 —— 挡住模型重复挑同一条追问
    const asks = wireCurrent.turns.map((x) => String(x.ask ?? ''))
    d = resolveTurn(out.parsed, detail.追问, detail.要点.length, asks)

    // 单篇过题的考点行没有人写的清单,靠模型现抽。**它三次里会空一次** ——
    // 空了就退化成 0/0 的无效判分,而且不报错。补一次定向重试。
    // 纠正消息只活在这次请求里,不写回转录(前缀缓存不受影响)。
    if (detail.selfPoints && d.points.length === 0) {
      const again = await chatJson<RawTurn>(
        'live',
        [
          ...built.messages,
          {
            role: 'user' as const,
            content:
              '你的 `要点` 是空的。这一题没有现成清单,**必须**先从参考答案原文里抽 3–5 条要点' +
              '写进 `要点` 字段,再用 `hit`/`miss` 的序号对应它们。重新输出完整 JSON。',
          },
        ],
        { maxTokens: 1200 },
      )
      const d2 = resolveTurn(again.parsed, detail.追问, detail.要点.length, asks)
      if (d2.points.length > 0) {
        d = d2
        res = again.res
      }
    }
  } catch (e) {
    return NextResponse.json({ error: `判卷失败:${(e as Error).message}` }, { status: 502 })
  }

  const article = corpus.articles.find((a) => a.title === detail.article)

  return NextResponse.json({
    hit: d.hit,
    miss: d.miss,
    next: d.next,
    followUp: d.followUp,
    source: d.source,
    project: d.project,
    // 判卷 JSON 原样回传,前端存进转录 —— 下一轮它会作为 assistant 消息进上下文
    verdict: res.content.trim(),
    // 过题档一题一结算;面试档一律不给
    ...(mode === 'drill'
      ? {
          要点: detail.要点.length ? detail.要点 : d.points,
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
