// 纯函数模块:不碰 fs,素材全部由调用方传入,单测可以喂固定数据
import { examTableBlock, materialBlock, systemPrompt, type Mode } from './prompts'
import type { Msg } from './llm'
import type { ExamPoint } from './types'

/**
 * 上下文规模的**软警戒线**。云端 1M 上下文,所以这不再是硬闸,
 * 只是超了要提醒 —— 真到几十万 token,该担心的是长上下文质量而不是显存。
 */
export const SOFT_LIMIT = 120_000

/**
 * 保守的 token 估算:**宁可高估**。CJK 按 1 字 1 token(实际约 1.5 字/token),
 * 其余按 3.5 字符 1 token。不引入真 tokenizer:对一个监控指标不值当。
 */
const CJK_RE = /[　-〿一-鿿＀-￯]/g

export function estimateTokens(text: string): number {
  if (!text) return 0
  const cjk = (text.match(CJK_RE) ?? []).length
  return Math.ceil(cjk + (text.length - cjk) / 3.5)
}

/** 一道题在本场里的完整痕迹 */
export interface QuestionRecord {
  /** 判分材料:要点 / 参考答案 / 追问池 / 考点表 */
  material: {
    题目: string
    要点: string[]
    答案: string
    追问: string[]
    needsReview: boolean
    examPoints: ExamPoint[]
    /**
     * 面试档的阶段名(开场 / 项目深挖 / 技术延伸 / 广度补充)。
     * **一道题定死一次就不能再变** —— 它进的是材料块,材料块一变前缀缓存就断。
     */
    phase?: string
    /** 阶段专属的额外说明(开场的项目清单等) */
    extra?: string
  }
  /** 这道题下面的问答轮次(主问 + 若干追问) */
  turns: { ask: string; answer: string; verdict?: string }[]
}

export interface SessionInput {
  mode: Mode
  toneSamples: string[]
  resumeName: string
  /** 简历**全文**,不是摘要 —— 云端装得下,面试官才能真的深挖项目 */
  resumeBody: string
  /**
   * 上下文来源。单篇过题时换成 `article` —— 第二条消息就从「候选人简历」
   * 变成「本次考察的文章全文」。**结构一个字不用改**,而模型手里有整篇原文,
   * 考点行的判分才站得住(那是这个场景能成立的关键)。
   */
  sourceKind?: 'resume' | 'article'
  /** 已经问完的题,按顺序 */
  history: QuestionRecord[]
  /** 当前这道题 */
  current: QuestionRecord
}

export interface BuiltContext {
  messages: Msg[]
  estimatedTokens: number
  /** 预计能命中前缀缓存的部分(= 最后一条 user 消息之前的全部) */
  cacheableTokens: number
  overSoftLimit: boolean
}

/**
 * 组装整场对话,**严格只追加**。
 *
 * 这是为 DeepSeek 的前缀缓存设计的:缓存命中价是未命中价的 1/30,
 * 所以只要每条消息一旦写下就永不修改,除最后一条外全部按 1/30 计费。
 *
 * ```
 * [system]     面试官人格 + 评分标准 + 真题口气           ← 整场不变
 * [user]       候选人简历全文                            ← 整场不变
 *   ↓ 每题追加,写下就不再动
 * [user]       第 N 题的判分材料(要点/答案/追问池/考点表)
 * [assistant]  题干(面试官问出口的话)
 * [user]       我的回答
 * [assistant]  判卷 JSON(模型自己上一轮的输出)
 *   ↓
 * [user]       我的回答                                  ← 唯一的新消息
 * ```
 *
 * **历史题的参考答案留在上下文里是安全的**:面试档下模型只能输出追问的**序号**,
 * 正文由我们查表得到(见 `resolveTurn`),它没有输出自由文本的通道。
 */
export function buildSession(input: SessionInput): BuiltContext {
  const messages: Msg[] = [
    { role: 'system', content: systemPrompt(input.mode, input.toneSamples) },
    {
      role: 'user',
      content:
        input.sourceKind === 'article'
          ? `## 本次考察的文章(${input.resumeName})\n\n${input.resumeBody}`
          : `## 候选人简历(${input.resumeName})\n\n${input.resumeBody}`,
    },
  ]

  const pushQuestion = (q: QuestionRecord, isCurrent: boolean) => {
    messages.push({
      role: 'user',
      content: [materialBlock(q.material), examTableBlock(q.material.examPoints)]
        .filter(Boolean)
        .join('\n\n'),
    })
    q.turns.forEach((t, i) => {
      messages.push({ role: 'assistant', content: t.ask })
      messages.push({ role: 'user', content: t.answer })
      // 最后一轮还没判卷 —— 它就是这次要模型回答的东西
      const isLast = isCurrent && i === q.turns.length - 1
      if (t.verdict && !isLast) messages.push({ role: 'assistant', content: t.verdict })
    })
  }

  for (const q of input.history) pushQuestion(q, false)
  pushQuestion(input.current, true)

  const total = messages.reduce((s, m) => s + estimateTokens(m.content), 0)
  const cacheable = messages
    .slice(0, -1)
    .reduce((s, m) => s + estimateTokens(m.content), 0)

  return {
    messages,
    estimatedTokens: total,
    cacheableTokens: cacheable,
    overSoftLimit: total > SOFT_LIMIT,
  }
}
