// 纯函数模块:不碰 fs,服务端与单测都能直接喂固定数据
import type { Candidate } from './types'

export interface SamplerProfile {
  /**
   * 第一级门禁(硬过滤):在册章节,如 ['04-Infra','01-模型结构']。
   * **来自简历 frontmatter 的 `章节:`,人工定,不由 LLM 推断**——
   * 漏写一个章节最多少考几道题,LLM 多推一个章节就是问你完全没做过的东西。
   */
  chapters: string[]
  /** 第二级(软权重):文章名 → 0–3 亲和度,由画像脚本产出。**0 是硬否决,不是降权** */
  affinity: Record<string, number>
  /** 画像里没提到的文章按几分算(没有画像时全部走这个),默认 1 = 岗位常识 */
  fallbackAffinity?: number
}

export interface SessionState {
  /** 本场已问过的候选 id */
  asked: ReadonlySet<string>
  /** 上一轮问的文章,用于避免连着问同一个 topic */
  lastArticle?: string
}

export const WEIGHTS = {
  /** 下标 = 亲和度;0 分权重为 0 → 直接出局 */
  affinity: [0, 1, 2, 3] as const,
  highfreq: 2,
  fromInterview: 1.5,
  mastered: 0.2,
  /** 已问过不是永久出局:池子过完一轮后还能再抽到 */
  asked: 0.05,
  sameArticle: 0.3,
  /** 知识库考点行没有 `## 要点`/`## 追问`,判卷素材比题库题少 */
  examPoint: 0.4,
} as const

export const EMPTY_SESSION: SessionState = { asked: new Set() }

function affinityOf(c: Candidate, profile: SamplerProfile): number {
  const raw = profile.affinity[c.article]
  return raw === undefined ? (profile.fallbackAffinity ?? 1) : raw
}

/**
 * 第一级门禁:章节不在册的,永远进不了候选集。
 * 这是防幻觉的硬闸——不是让模型「尽量别问」,是根本不给它候选。
 */
export function eligible(pool: readonly Candidate[], profile: SamplerProfile): Candidate[] {
  const allowed = new Set(profile.chapters)
  return pool.filter((c) => allowed.has(c.chapter) && affinityOf(c, profile) > 0)
}

/** 单个候选的权重;返回 0 表示不可抽 */
export function weightOf(
  c: Candidate,
  profile: SamplerProfile,
  session: SessionState = EMPTY_SESSION,
): number {
  const aff = affinityOf(c, profile)
  const tier = WEIGHTS.affinity[Math.max(0, Math.min(3, Math.round(aff)))]
  if (tier === 0) return 0
  let w = tier
  if (c.highfreq) w *= WEIGHTS.highfreq
  if (c.fromInterview) w *= WEIGHTS.fromInterview
  if (c.mastered) w *= WEIGHTS.mastered
  if (session.asked.has(c.id)) w *= WEIGHTS.asked
  if (session.lastArticle && c.article === session.lastArticle) w *= WEIGHTS.sameArticle
  if (c.kind === 'exam-point') w *= WEIGHTS.examPoint
  return w
}

/**
 * 加权随机抽一道。`rnd` 注入是为了单测可复现。
 * 池子空(章节全不在册,或亲和度全 0)时返回 null,**绝不退化成随机乱抽**。
 */
export function pick(
  pool: readonly Candidate[],
  profile: SamplerProfile,
  session: SessionState = EMPTY_SESSION,
  rnd: () => number = Math.random,
): Candidate | null {
  const cands = eligible(pool, profile)
  if (cands.length === 0) return null
  const weights = cands.map((c) => weightOf(c, profile, session))
  const total = weights.reduce((s, w) => s + w, 0)
  if (total <= 0) return null
  let r = rnd() * total
  for (let i = 0; i < cands.length; i++) {
    r -= weights[i]
    if (r < 0) return cands[i]
  }
  return cands[cands.length - 1]
}

/** 可复现的伪随机源(单测与「重放某场面试」都用它) */
export function seededRandom(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    // mulberry32
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
