/**
 * OpenAI 兼容客户端。三个阶段各自可配 —— 换后端只改环境变量,不改代码。
 *
 * 为什么按阶段分:延迟敏感度差三个数量级。
 *   live 每轮都跑、必须够快 → 关思考、极简输出
 *   prep/post 一场一次、你在读东西 → 可以慢,也可以指向更强的模型
 */
export type Phase = 'live' | 'prep' | 'post'

export interface Msg {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface LlmConfig {
  baseUrl: string
  model: string
  apiKey: string
  timeoutMs: number
}

const DEFAULTS: LlmConfig = {
  baseUrl: 'http://localhost:1234/v1',
  model: 'qwen3-14b-mlx',
  apiKey: 'lm-studio', // LM Studio 不校验,但 OpenAI 客户端要求非空
  timeoutMs: 120_000,
}

function envFor(phase: Phase, key: string): string | undefined {
  // 阶段专属优先,回落到全局:INTERVIEW_LLM_POST_MODEL → INTERVIEW_LLM_MODEL
  return (
    process.env[`INTERVIEW_LLM_${phase.toUpperCase()}_${key}`] ??
    process.env[`INTERVIEW_LLM_${key}`]
  )
}

export function configFor(phase: Phase): LlmConfig {
  const timeout = envFor(phase, 'TIMEOUT_MS')
  return {
    baseUrl: envFor(phase, 'BASE_URL') ?? DEFAULTS.baseUrl,
    model: envFor(phase, 'MODEL') ?? DEFAULTS.model,
    apiKey: envFor(phase, 'API_KEY') ?? DEFAULTS.apiKey,
    timeoutMs: timeout ? Number(timeout) : DEFAULTS.timeoutMs,
  }
}

/**
 * 关思考的写法各家不同,而且**都会静默失败**:参数不认就当没写,只表现为「怎么这么慢」。
 * 所以两种写法一起上,谁认谁生效。
 *   DeepSeek  → `thinking: {type:"disabled"}`(v4 全系默认开思考)
 *   LM Studio → 消息末尾 ` /no_think`(它**不支持** chat_template_kwargs,实测收下但无效)
 *   百炼      → `enable_thinking: false`
 *
 * 实测:判卷这类「对着清单逐条比对」的匹配任务,开思考更慢更贵还更差
 * (DeepSeek 上 1.95s→19.19s,而且思考 3546 字撑爆输出预算导致 JSON 截断)。
 */
const NO_THINK_SUFFIX = ' /no_think'

function noThinkBody(): Record<string, unknown> {
  return { thinking: { type: 'disabled' }, enable_thinking: false }
}

export interface ChatResult {
  content: string
  promptTokens: number
  /** 命中前缀缓存的输入 token(DeepSeek 按 1/30 计价) */
  cachedTokens: number
  /** 不含思考的正文 token */
  outputTokens: number
  reasoningTokens: number
  elapsedMs: number
}

export async function chat(
  phase: Phase,
  messages: Msg[],
  opts: { maxTokens?: number; temperature?: number; noThink?: boolean } = {},
): Promise<ChatResult> {
  const cfg = configFor(phase)
  const noThink = opts.noThink !== false
  const sent = noThink
    ? messages.map((m, i) =>
        i === messages.length - 1 ? { ...m, content: m.content + NO_THINK_SUFFIX } : m,
      )
    : messages

  const started = Date.now()
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), cfg.timeoutMs)
  try {
    const res = await fetch(`${cfg.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${cfg.apiKey}` },
      signal: ctrl.signal,
      body: JSON.stringify({
        model: cfg.model,
        messages: sent,
        max_tokens: opts.maxTokens ?? 400,
        temperature: opts.temperature ?? 0,
        ...(noThink ? noThinkBody() : {}),
      }),
    })
    if (!res.ok) {
      throw new Error(`LLM ${res.status}: ${(await res.text()).slice(0, 300)}`)
    }
    const data = await res.json()
    const usage = data.usage ?? {}
    const reasoning =
      usage.completion_tokens_details?.reasoning_tokens ??
      (data.choices?.[0]?.message?.reasoning_content ? 1 : 0)
    const content = data.choices?.[0]?.message?.content ?? ''
    if (!content.trim()) {
      throw new Error(
        `模型返回空正文(finish_reason=${data.choices?.[0]?.finish_reason},思考 ${reasoning})。` +
          `多半是思考把输出预算烧光了 —— 检查关思考的参数有没有生效。`,
      )
    }
    return {
      content,
      promptTokens: usage.prompt_tokens ?? 0,
      cachedTokens: usage.prompt_cache_hit_tokens ?? usage.prompt_tokens_details?.cached_tokens ?? 0,
      outputTokens: (usage.completion_tokens ?? 0) - reasoning,
      reasoningTokens: reasoning,
      elapsedMs: Date.now() - started,
    }
  } catch (e) {
    if ((e as Error).name === 'AbortError') {
      throw new Error(`LLM 调用超时(${cfg.timeoutMs}ms)@ ${cfg.baseUrl}`)
    }
    throw e
  } finally {
    clearTimeout(timer)
  }
}

/**
 * 从模型输出里抠 JSON。小模型偶尔会裹 ```json 或在前面加一句废话,
 * 所以不能直接 JSON.parse ——取第一个 `{` 到最后一个 `}`。
 */
export function extractJson<T>(raw: string): T {
  const cleaned = raw.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim()
  const start = cleaned.indexOf('{')
  const end = cleaned.lastIndexOf('}')
  if (start < 0 || end <= start) {
    throw new Error(`模型没有输出 JSON:${raw.slice(0, 200)}`)
  }
  return JSON.parse(cleaned.slice(start, end + 1)) as T
}

/** 模型返回的极简判卷结果 */
export interface RawTurn {
  hit?: number[]
  miss?: number[]
  next?: string
  /** 现成追问的序号;-1 表示自己写 */
  fu?: number
  /** fu = -1 时模型自己写的追问(也用于简历深挖) */
  ask?: string
  /** 给复盘留的一句话,候选人看不到 */
  note?: string
  评价?: string
}

export interface TurnDecision {
  hit: number[]
  miss: number[]
  /** `end` 是模型主动收尾 —— 面试节奏由它主导,不由轮次上限决定 */
  next: 'followup' | 'nextq' | 'end'
  /** 最终要问出口的话 */
  followUp: string
  /** 追问来自哪:池子里的第几条,还是模型生成的 */
  source: { kind: 'pool'; index: number } | { kind: 'generated' } | { kind: 'none' }
  comment?: string
  /** 面试官的私下备注,进复盘 */
  note?: string
}

/**
 * 把模型的极简输出还原成完整决策。
 * **追问正文由我们查表得到,不信模型复述** —— 模型只给序号,越界就退化成换题。
 * 这同时让「上下文里堆着历史参考答案」变得安全:面试档里模型根本没有输出自由文本的通道。
 */
export function resolveTurn(raw: RawTurn, followUps: string[], pointCount: number): TurnDecision {
  const inRange = (n: unknown) => typeof n === 'number' && n >= 0 && n < pointCount
  const hit = (raw.hit ?? []).filter(inRange)
  const miss = (raw.miss ?? []).filter(inRange).filter((i) => !hit.includes(i))
  const note = typeof raw.note === 'string' ? raw.note.trim() || undefined : undefined

  if (raw.next === 'end') {
    return { hit, miss, next: 'end', followUp: '', source: { kind: 'none' }, comment: raw.评价, note }
  }
  if (raw.next === 'nextq') {
    return { hit, miss, next: 'nextq', followUp: '', source: { kind: 'none' }, comment: raw.评价, note }
  }
  const idx = raw.fu
  if (typeof idx === 'number' && idx >= 0 && idx < followUps.length) {
    return {
      hit,
      miss,
      next: 'followup',
      followUp: followUps[idx],
      source: { kind: 'pool', index: idx },
      comment: raw.评价,
      note,
    }
  }
  const written = (raw.ask ?? '').trim()
  if (written) {
    return { hit, miss, next: 'followup', followUp: written, source: { kind: 'generated' }, comment: raw.评价, note }
  }
  // 序号越界又没写正文 —— 不猜,直接换题
  return { hit, miss, next: 'nextq', followUp: '', source: { kind: 'none' }, comment: raw.评价, note }
}
