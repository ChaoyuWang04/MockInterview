import { NextResponse } from 'next/server'
import { buildCorpus, loadQuestionDetail } from '@/lib/interview/corpus'
import { listResumes, loadProfile, samplerProfileFor } from '@/lib/interview/resume'
import { pick, seededRandom } from '@/lib/interview/sampler'

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

  const resume = listResumes().find((r) => r.slug === resumeSlug)
  if (!resume) return NextResponse.json({ error: `没有这份简历:${resumeSlug}` }, { status: 400 })

  const corpus = buildCorpus()
  // M0 只从题库题里抽:它们自带 `## 要点`(评分细则)与 `## 追问`(现成追问池)。
  // 知识库考点行没有这两样,判卷会退化成主观打分,留到后面单独处理。
  const pool = corpus.candidates.filter((c) => c.kind === 'question')
  const profile = samplerProfileFor(resume, loadProfile(resume.slug))

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
    面经: chosen.fromInterview,
    待校对: detail.entry.needsReview,
    poolSize: pool.length,
    // ⚠️ 故意不返回 `答案` 与 `要点` 正文 —— 面试档里它们不能出现在浏览器
  })
}
