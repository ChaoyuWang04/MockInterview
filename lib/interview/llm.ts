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
  opts: { maxTokens?: number; temperature?: number; noThink?: boolean; jsonMode?: boolean } = {},
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
        // 服务端强制合法 JSON。**不进 messages,所以前缀缓存不受影响**。
        // 不认这个字段的后端会忽略它,那时靠 chatJson 的重试兜底。
        ...(opts.jsonMode ? { response_format: { type: 'json_object' } } : {}),
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
      const why =
        data.choices?.[0]?.finish_reason === 'length'
          ? `预算烧光了(max_tokens=${opts.maxTokens ?? 400},**思考与正文共用这一份预算**)。` +
            (noThink ? '关思考的参数可能没生效。' : '这条是故意开思考的,把 maxTokens 调大。')
          : '不是长度问题 —— 开了 JSON 模式时偶发,由 chatJson 的重试兜底。'
      throw new Error(
        `模型返回空正文(finish_reason=${data.choices?.[0]?.finish_reason},思考 ${reasoning})。${why}`,
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

/**
 * 要 JSON 的那种调用:开服务端 JSON 模式,解析失败再补一次纠正重试。
 *
 * 存在的理由是实测踩到的:模型偶尔会**把追问当纯文本直接回出来**,不裹 JSON
 * (日志里两次都是想问简历那种自由追问时发生的)。之前这会让 `extractJson` 抛异常、
 * 整条路由崩掉、浏览器只看到「Unexpected end of JSON input」。
 *
 * ⚠️ 纠正消息**只在这一次请求里存在,绝不写回转录**。写回去就等于修改了已写下的消息,
 * 前缀缓存会从那一条起全部失效 —— 不报错、不变慢,只是悄悄贵 30 倍(见 context.ts 的红线)。
 */
export async function chatJson<T>(
  phase: Phase,
  messages: Msg[],
  opts: { maxTokens?: number; temperature?: number } = {},
): Promise<{ parsed: T; res: ChatResult }> {
  // ⚠️ 第一次调用**必须也在 try 里**。写在外面的版本有个真实的坑:
  // DeepSeek 开 JSON 模式时偶尔返回空正文(finish_reason=stop、思考 0),
  // 这个异常是 `chat()` 抛的,不是 `extractJson` 抛的 —— 兜底重试压根不会触发,
  // 整轮直接 502。实测同样的开场白,有的种子过、有的种子挂。
  try {
    const res = await chat(phase, messages, { ...opts, jsonMode: true })
    return { parsed: extractJson<T>(res.content), res }
  } catch (first) {
    // 重试**关掉 JSON 模式** —— 空正文本身多半就是 JSON 模式引起的,
    // 再开一次大概率还是空。裹 markdown 或加废话都由 extractJson 兜住。
    const nudge: Msg[] = [
      ...messages,
      {
        role: 'user',
        content:
          '你刚才没有按格式输出。只输出一段 JSON,不要任何解释、不要 markdown 代码块。' +
          '追问不要写成正文 —— 现成追问填 `fu` 序号,自己写的追问放进 `ask` 字段。',
      },
    ]
    try {
      const retry = await chat(phase, nudge, { ...opts, temperature: 0 })
      return { parsed: extractJson<T>(retry.content), res: retry }
    } catch (second) {
      // 两次都失败,把两次的原因都带出去 —— 只报第二次会掩盖首因
      throw new Error(`${(first as Error).message};重试后仍失败:${(second as Error).message}`)
    }
  }
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
  /** 想深挖哪个项目(面试档;越界或缺省时由代码随机兜底) */
  proj?: number
  /** 给复盘留的一句话,候选人看不到 */
  note?: string
  /**
   * 模型现抽的要点清单。**只在没有人写的清单时用**(单篇过题的考点行),
   * 模型从参考答案原文里抽 3–5 条,再拿它当 hit/miss 的下标基准。
   */
  要点?: string[]
  评价?: string
}

export interface TurnDecision {
  hit: number[]
  miss: number[]
  /**
   * `end` 是模型主动收尾 —— 面试节奏由它主导,不由轮次上限决定。
   * `nextphase` 只在面试档有意义:模型觉得这个阶段问透了。**阶段的轮次上限仍由代码卡死**,
   * 模型只能提前结束、不能拖长(见 phases.ts 的 PHASE_CAPS)。
   */
  next: 'followup' | 'nextq' | 'nextphase' | 'end'
  /** 最终要问出口的话 */
  followUp: string
  /** 追问来自哪:池子里的第几条,还是模型生成的 */
  source: { kind: 'pool'; index: number } | { kind: 'generated' } | { kind: 'none' }
  comment?: string
  /** 面试官的私下备注,进复盘 */
  note?: string
  /**
   * 模型选中的项目序号。**只是建议** —— 越界、重复、或压根没填时,
   * 前端会用会话种子随机挑一个还没问过的。模型不配合不能让面试卡住。
   */
  project?: number
  /** 模型现抽的要点清单;人写的清单存在时为空数组 */
  points: string[]
}

/**
 * 把模型的极简输出还原成完整决策。
 * **追问正文由我们查表得到,不信模型复述** —— 模型只给序号,越界就退化成换题。
 * 这同时让「上下文里堆着历史参考答案」变得安全:面试档里模型根本没有输出自由文本的通道。
 */
export function resolveTurn(
  raw: RawTurn,
  followUps: string[],
  pointCount: number,
  /**
   * 这道题下面已经问出口的话。**用来挡重复追问**。
   *
   * 项目深挖阶段的追问池只有一条(画像里那个「可问」),模型反复挑 `fu: 0` 时,
   * 候选人会一字不差地听到同一个问题两三遍 —— 实测踩到。挡住之后退化成
   * 模型自己写的追问,它也重复或没写就换题/转阶段。
   */
  askedBefore: readonly string[] = [],
): TurnDecision {
  // 没有人写的清单时(单篇过题的考点行,pointCount = 0),用模型现抽的那份当基准。
  // 不这么做的话 inRange 会把所有 hit/miss 全过滤掉 —— 判分静默变成 0/0。
  const points = pointCount > 0 ? [] : (raw.要点 ?? []).map(String).map((s) => s.trim()).filter(Boolean).slice(0, 8)
  const scale = pointCount > 0 ? pointCount : points.length
  const inRange = (n: unknown) => typeof n === 'number' && n >= 0 && n < scale
  const hit = (raw.hit ?? []).filter(inRange)
  const miss = (raw.miss ?? []).filter(inRange).filter((i) => !hit.includes(i))
  const note = typeof raw.note === 'string' ? raw.note.trim() || undefined : undefined
  // 只做类型与非负校验;越不越界由前端按实际项目数判断(这里不知道有几个项目)
  const project = typeof raw.proj === 'number' && raw.proj >= 0 ? Math.floor(raw.proj) : undefined

  if (raw.next === 'end') {
    return { hit, miss, next: 'end', followUp: '', source: { kind: 'none' }, comment: raw.评价, note, project, points }
  }
  if (raw.next === 'nextq') {
    return { hit, miss, next: 'nextq', followUp: '', source: { kind: 'none' }, comment: raw.评价, note, project, points }
  }
  if (raw.next === 'nextphase') {
    return { hit, miss, next: 'nextphase', followUp: '', source: { kind: 'none' }, comment: raw.评价, note, project, points }
  }
  const said = new Set(askedBefore.map((s) => s.trim()).filter(Boolean))
  const idx = raw.fu
  if (typeof idx === 'number' && idx >= 0 && idx < followUps.length && !said.has(followUps[idx].trim())) {
    return {
      hit,
      miss,
      next: 'followup',
      followUp: followUps[idx],
      source: { kind: 'pool', index: idx },
      comment: raw.评价,
      note,
      project,
      points,
    }
  }
  const written = (raw.ask ?? '').trim()
  if (written && !said.has(written)) {
    return { hit, miss, next: 'followup', followUp: written, source: { kind: 'generated' }, comment: raw.评价, note, project, points }
  }
  // 序号越界又没写正文 —— 不猜,直接换题
  return { hit, miss, next: 'nextq', followUp: '', source: { kind: 'none' }, comment: raw.评价, note, project, points }
}
