// 纯函数模块:不碰 fs,服务端与单测都能直接喂固定数据
import type { Candidate } from './types'

export interface SamplerProfile {
  /**
   * 第一级门禁(硬过滤):在册章节,如 ['04-Infra','01-模型结构']。
   * **来自简历 frontmatter 的 `章节:`,人工定,不由 LLM 推断**——
   * 漏写一个章节最多少考几道题,LLM 多推一个章节就是问你完全没做过的东西。
   */
  chapters: string[]
  /**
   * 章节权重,同样来自简历 frontmatter。**决定「这场面试有多少比例问这个方向」**。
   *
   * 存在的理由:权重若逐个候选计算,章节的中签率会正比于它有多少语料,
   * 使内容多的章节压过简历真正关注的方向。
   * 现在改成两级采样:**先按这里的权重选章节,再在章节内部选题** —— 章节有多少题
   * 不再影响它被选中的概率。缺省(简历只写了章节列表)时全部按 1,即各章均等。
   */
  chapterWeights?: Record<string, number>
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
  if (c.mastered) w *= WEIGHTS.mastered
  if (session.asked.has(c.id)) w *= WEIGHTS.asked
  if (session.lastArticle && c.article === session.lastArticle) w *= WEIGHTS.sameArticle
  if (c.kind === 'exam-point') w *= WEIGHTS.examPoint
  return w
}

/** 从一组候选里按权重抽一个;全零返回 null */
function weightedPick(
  cands: readonly Candidate[],
  weights: readonly number[],
  rnd: () => number,
): Candidate | null {
  const total = weights.reduce((s, w) => s + w, 0)
  if (total <= 0) return null
  let r = rnd() * total
  for (let i = 0; i < cands.length; i++) {
    r -= weights[i]
    if (r < 0) return cands[i]
  }
  return cands[cands.length - 1]
}

/**
 * 抽一道。**两级采样:先选章节,再在章节内选题。**`rnd` 注入是为了单测可复现。
 *
 * 为什么必须分两级见 `SamplerProfile.chapterWeights` 的注释 —— 一级采样下
 * 章节的中签率正比于语料量,简历权重会被语料分布淹没。
 *
 * 章节的实际权重 = 简历给的权重 × (该章还有没有能抽的题)。乘上后者是为了
 * 让「本章的题在这场里问完了」自然降权到几乎不选,而不是空转。
 *
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

  // 按章节分桶,顺带算出每桶内部的权重
  const buckets = new Map<string, { cands: Candidate[]; weights: number[]; sum: number }>()
  for (const c of cands) {
    const w = weightOf(c, profile, session)
    if (w <= 0) continue
    const b = buckets.get(c.chapter) ?? { cands: [], weights: [], sum: 0 }
    b.cands.push(c)
    b.weights.push(w)
    b.sum += w
    buckets.set(c.chapter, b)
  }
  if (buckets.size === 0) return null

  const chapters = [...buckets.keys()]
  const cw = profile.chapterWeights
  const chapterWeights = chapters.map((ch) => {
    const base = cw?.[ch] ?? 1
    if (base <= 0) return 0
    // 桶内权重和只用来判「这章还剩多少可问的」:已问过的题权重被压到 0.05,
    // 一章问光了它的和会趋近 0,自然让位给别的章。不直接用和本身,
    // 否则又变回「语料多的章赢」——所以做归一化,封顶 1。
    const remaining = Math.min(1, b_sum(buckets, ch) / Math.max(1, buckets.get(ch)!.cands.length))
    return base * Math.max(0.05, remaining)
  })

  const chapter = chapters[pickIndex(chapterWeights, rnd)] ?? chapters[0]
  const b = buckets.get(chapter)!
  return weightedPick(b.cands, b.weights, rnd)
}

function b_sum(buckets: Map<string, { sum: number }>, ch: string): number {
  return buckets.get(ch)?.sum ?? 0
}

function pickIndex(weights: readonly number[], rnd: () => number): number {
  const total = weights.reduce((s, w) => s + w, 0)
  if (total <= 0) return 0
  let r = rnd() * total
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i]
    if (r < 0) return i
  }
  return weights.length - 1
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
