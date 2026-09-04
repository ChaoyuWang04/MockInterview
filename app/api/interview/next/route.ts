import { NextResponse } from 'next/server'
import { buildCorpus, loadQuestionDetail } from '@/lib/interview/corpus'
import { listResumes, loadProfile, samplerProfileFor } from '@/lib/interview/resume'
import { pick, seededRandom } from '@/lib/interview/sampler'
import { examPointMaterial } from '@/lib/interview/kbdrill'
import {
  INTRO_ID,
  PHASE_LABEL,
  introMaterial,
  projectCount,
  projectId,
  projectMaterial,
  relatedTopics,
  type InterviewPhase,
} from '@/lib/interview/phases'

export const dynamic = 'force-dynamic'

/**
 * 抽下一题。**服务端无状态** —— 本场已问清单由前端持有并回传,
 * 这样刷新页面、重建服务都不会丢会话,也不用引入任何会话存储。
 */
export async function POST(req: Request) {
  let body: Record<string, unknown>
  try {
    body = (await req.json()) as Record<string, unknown>
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }

  const resumeSlug = String(body.resume ?? '')
  const asked = Array.isArray(body.asked) ? body.asked.map(String) : []
  const lastArticle = typeof body.lastArticle === 'string' ? body.lastArticle : undefined
  const seed = typeof body.seed === 'number' ? body.seed : Date.now()

  // ── 单篇过题:池子只限这一篇,不看简历门禁也不看亲和度(是你主动点进来的)──
  const article = typeof body.article === 'string' ? body.article : ''
  if (article) {
    const corpus0 = buildCorpus()
    const entry = corpus0.articles.find((a) => a.title === article)
    if (!entry) return NextResponse.json({ error: `没有这篇文章:${article}` }, { status: 400 })
    // ⚠️ **不能直接筛 corpus.candidates**。`buildCorpus` 里有一句去重:
    // 有题库题的文章,它的考点行根本不进候选池(避免通用面试里同一篇问两遍)。
    // 单篇过题要的正好相反 —— 全过一遍。所以这里自己组池:题库题 + 该篇**全部**考点行。
    const mine = [
      ...corpus0.candidates.filter((c) => c.kind === 'question' && c.article === article),
      ...entry.examPoints.map((p, i) => ({
        id: `k:${article}#${i}`,
        kind: 'exam-point' as const,
        chapter: entry.chapter,
        article,
        ask: p.ask,
        highfreq: false,
        mastered: false,
      })),
    ]
    const left = mine.filter((c) => !asked.includes(c.id))
    // 池子过完 = 这一篇考完了,前端据此收尾并生成复盘
    if (left.length === 0) return NextResponse.json({ error: 'ARTICLE_DONE', article }, { status: 409 })
    // 纯随机取一个还没问过的:单篇不需要加权,你要的是「全部过一遍」
    const chosen0 = left[Math.floor(seededRandom(seed)() * left.length)]
    const detail0 =
      chosen0.kind === 'question'
        ? loadQuestionDetail(chosen0.id.slice(2))
        : examPointMaterial(article, Number(chosen0.id.split('#')[1]))
    if (!detail0) return NextResponse.json({ error: `材料读不出来:${chosen0.id}` }, { status: 500 })
    const 要点 = 'entry' in detail0 ? detail0.要点 : detail0.要点
    return NextResponse.json({
      id: chosen0.id,
      article,
      chapter: entry.chapter,
      题目: detail0.题目,
      要点数: 要点.length,
      追问数: detail0.追问.length,
      highfreq: chosen0.highfreq,
      真题: chosen0.kind === 'question',
      待校对: 'entry' in detail0 ? detail0.entry.needsReview : false,
      poolSize: mine.length,
      remaining: left.length,
    })
  }

  const resume = listResumes().find((r) => r.slug === resumeSlug)
  if (!resume) return NextResponse.json({ error: `没有这份简历:${resumeSlug}` }, { status: 400 })

  const rawProfile = loadProfile(resume.slug)
  const phase = String(body.phase ?? 'breadth') as InterviewPhase
  const projectIndex = typeof body.projectIndex === 'number' ? body.projectIndex : 0
  // 会话种子:整场不变,决定项目清单的呈现顺序。见 phases.ts 的 projectOrder
  const sessionSeed = typeof body.sessionSeed === 'number' ? body.sessionSeed : 0

  // ── 阶段题:自我介绍与项目深挖不从题库抽,材料由简历与画像直接给 ──
  if (phase === 'intro') {
    const m = introMaterial(rawProfile, sessionSeed)
    return NextResponse.json({
      id: INTRO_ID,
      article: m.article,
      chapter: m.chapter,
      题目: m.题目,
      要点数: 0,
      追问数: 0,
      highfreq: false,
      真题: false,
      待校对: false,
      poolSize: 0,
      phase,
      phaseLabel: PHASE_LABEL[phase],
      projectCount: projectCount(rawProfile),
    })
  }
  if (phase === 'project') {
    const m = projectMaterial(resume, rawProfile, projectIndex)
    // 没有画像 / 项目问完了 → 让前端直接进下一阶段,而不是空转
    if (!m) return NextResponse.json({ error: 'NO_MORE_PROJECTS', phase }, { status: 409 })
    return NextResponse.json({
      id: projectId(projectIndex),
      article: m.article,
      chapter: m.chapter,
      题目: m.题目,
      要点数: m.要点.length,
      追问数: m.追问.length,
      highfreq: false,
      真题: false,
      待校对: false,
      poolSize: projectCount(rawProfile),
      phase,
      phaseLabel: PHASE_LABEL[phase],
      projectCount: projectCount(rawProfile),
    })
  }

  const corpus = buildCorpus()
  // 只从题库题里抽:它们自带 `## 要点`(评分细则)与 `## 追问`(现成追问池)。
  // 知识库考点行没有这两样,判卷会退化成主观打分,留到后面单独处理。
  let pool = corpus.candidates.filter((c) => c.kind === 'question')

  // 技术延伸阶段:只从**当前项目关联到的文章**里出题 —— 这是让技术题
  // 从他刚讲的项目里长出来的那座桥。
  //
  // ⚠️ 关联不到题时**直接跳过这个阶段,绝不退回全池**。退回全池的版本写过,
  // 表现是「技术延伸」阶段问出一堆和刚才那个项目毫无关系的题(实测:项目讲的是
  // GRPO 训练,却被问 GEMM tune 和 RAG 检索),而页面上还标着「技术延伸」——
  // 典型的不报错但行为不对。宁可少一个阶段,也不要假装它接上了。
  if (phase === 'tech') {
    const topics = new Set(relatedTopics(rawProfile, projectIndex))
    const narrowed = pool.filter((c) => topics.has(c.article))
    if (narrowed.length === 0) {
      return NextResponse.json(
        { error: 'NO_RELATED_QUESTIONS', phase, topics: [...topics] },
        { status: 409 },
      )
    }
    pool = narrowed
  }

  const profile = samplerProfileFor(resume, rawProfile)
  const chosen = pick(pool, profile, { asked: new Set(asked), lastArticle }, seededRandom(seed))
  if (!chosen) {
    return NextResponse.json(
      { error: '候选池是空的:简历的「章节」可能都不在册,或亲和度全被设成 0' },
      { status: 409 },
    )
  }

  const detail = loadQuestionDetail(chosen.id.slice(2))
  if (!detail) return NextResponse.json({ error: `题目读不出来:${chosen.id}` }, { status: 500 })

  return NextResponse.json({
    id: chosen.id,
    article: chosen.article,
    chapter: chosen.chapter,
    // 题干念原文,不改写:既保留真题口气,后面 TTS 缓存也能 100% 命中
    题目: detail.题目,
    要点数: detail.要点.length,
    追问数: detail.追问.length,
    highfreq: chosen.highfreq,
    真题: true,
    待校对: detail.entry.needsReview,
    poolSize: pool.length,
    phase,
    phaseLabel: PHASE_LABEL[phase] ?? '',
    projectCount: projectCount(rawProfile),
    // ⚠️ 故意不返回 `答案` 与 `要点` 正文 —— 面试档里它们不能出现在浏览器
  })
}
