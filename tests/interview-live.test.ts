/**
 * 真实 LLM 调用的验证。**默认跳过** —— `npm test` 必须保持秒级且不依赖外部服务
 * (仓库铁律:验证只跑 npm test)。要跑它:
 *
 *     npm run interview:live
 *
 * 后端由环境变量决定(见 .env.local),默认 DeepSeek。
 */
import { describe, it, expect } from 'vitest'
import { buildCorpus, loadQuestionDetail, loadToneSamples } from '../lib/interview/corpus'
import { buildSession } from '../lib/interview/context'
import type { QuestionRecord, SessionInput } from '../lib/interview/context'
import { chat, configFor, extractJson, resolveTurn } from '../lib/interview/llm'
import type { RawTurn } from '../lib/interview/llm'
import { listResumes } from '../lib/interview/resume'
import type { Mode } from '../lib/interview/prompts'

const LIVE = process.env.INTERVIEW_LIVE === '1'

/** 一轮判卷的延迟门禁。真面试官本来就会停顿两三秒,这个数按体感定,不按理论。 */
const LIVE_BUDGET_MS = 4000
const DRILL_BUDGET_MS = 6000

const EMPTY_ANSWER = '这道题我只知道一部分,先说这么多吧。'
const FULL_ANSWER =
  'GPU 内存层级从快到慢是寄存器、shared memory、L1、L2、HBM。能显式编程的只有寄存器和 shared memory,' +
  'L1/L2 是硬件自动管理的只能间接影响。shared memory 是 tiling 的地基——把数据从 HBM 搬进来让 block 内反复复用,' +
  '把多次慢访问变成一次慢访问加多次快访问。提高命中率四条:合并访存让 warp 内 32 线程访问连续地址、' +
  'tiling 提高复用、缩小工作集让热点装进 40MB 的 L2、按访问顺序调数据布局。' +
  '从 HBM 读一个数 400-600 周期而一次乘加只要几个周期,所以多数算子访存受限。' +
  'GPU 的 cache 主要用来提高带宽利用率,掩盖延迟靠线程多而不是缓存大。'

describe.skipIf(!LIVE)('真实 LLM 调用', () => {
  const corpus = buildCorpus()
  const tone = loadToneSamples()
  const resume = listResumes().find((r) => r.slug === '02-infra')!
  const targetId = corpus.candidates.find((c) => c.id.startsWith('q:AI Infra/001'))!.id.slice(2)
  const detail = loadQuestionDetail(targetId)!

  function record(id: string, answers: string[]): QuestionRecord {
    const d = loadQuestionDetail(id)!
    return {
      material: {
        题目: d.题目,
        要点: d.要点,
        答案: d.答案,
        追问: d.追问,
        needsReview: d.entry.needsReview,
        examPoints: corpus.articles.find((a) => a.title === d.entry.article)?.examPoints ?? [],
      },
      turns: answers.map((a, i) => ({ ask: i === 0 ? d.题目 : `追问 ${i}`, answer: a })),
    }
  }

  const session = (answer: string, mode: Mode = 'interview'): SessionInput => ({
    mode,
    toneSamples: tone,
    resumeName: resume.name,
    resumeBody: resume.body,
    history: [],
    current: record(targetId, [answer]),
  })

  async function ask(input: SessionInput, maxTokens = 300) {
    const built = buildSession(input)
    const r = await chat('live', built.messages, { maxTokens })
    return { r, built, d: resolveTurn(extractJson<RawTurn>(r.content), detail.追问, detail.要点.length) }
  }

  it('后端可达,配置读到了', async () => {
    const cfg = configFor('live')
    console.log(`  后端 ${cfg.baseUrl} · 模型 ${cfg.model}`)
    const r = await chat('live', [{ role: 'user', content: '只回答两个字:就绪' }], { maxTokens: 20 })
    expect(r.content.length).toBeGreaterThan(0)
  }, 120_000)

  it(`面试档一轮 ≤ ${LIVE_BUDGET_MS}ms,且思考确实关掉了`, async () => {
    const { r, built } = await ask(session(FULL_ANSWER))
    console.log(
      `  面试档 ${r.elapsedMs}ms | 输入 ${r.promptTokens}(缓存 ${r.cachedTokens})| 输出 ${r.outputTokens} | 思考 ${r.reasoningTokens}`,
    )
    console.log(`  上下文估算 ${built.estimatedTokens} tok,可缓存 ${built.cacheableTokens}`)
    expect(r.reasoningTokens).toBeLessThan(5)
    expect(r.elapsedMs).toBeLessThanOrEqual(LIVE_BUDGET_MS)
  }, 120_000)

  /**
   * ★ 判卷有没有真在判的**唯一**证据。
   * 踩过的坑:schema 示例里写了具体序号,模型原样抄回来 —— 空洞回答和完整回答
   * 输出一字不差,而当时的断言(hit+miss > 0)照样通过。
   */
  it('判卷真的依赖回答内容:空洞 0 命中,完整多数命中', async () => {
    const empty = await ask(session(EMPTY_ANSWER))
    const full = await ask(session(FULL_ANSWER))
    console.log(`  空洞 hit=${JSON.stringify(empty.d.hit)} / 完整 hit=${JSON.stringify(full.d.hit)}`)
    expect(empty.d.hit).toEqual([])
    expect(full.d.hit.length).toBeGreaterThan(detail.要点.length / 2)
  }, 200_000)

  it('追问来自人工写好的池子,不是模型编的', async () => {
    const { d } = await ask(session(FULL_ANSWER))
    if (d.next === 'followup') {
      expect(d.source.kind).toBe('pool')
      expect(detail.追问).toContain(d.followUp)
      console.log(`  追问 → ${d.followUp}`)
    }
  }, 120_000)

  it(`过题档带评价 ≤ ${DRILL_BUDGET_MS}ms`, async () => {
    const { r, d } = await ask(session(FULL_ANSWER, 'drill'), 400)
    console.log(`  过题档 ${r.elapsedMs}ms | 评价:${d.comment ?? '(无)'}`)
    expect(d.comment).toBeTruthy()
    expect(r.elapsedMs).toBeLessThanOrEqual(DRILL_BUDGET_MS)
  }, 120_000)

  /**
   * 泄漏防护是**结构性**的:面试档下模型只输出追问的序号,正文由我们查表得到。
   * 所以真正要断言的是「问出口的话必定来自人工写好的池子」,
   * 而不是去匹配某个字符串 —— 追问池里本来就可能含有和答案相同的数字。
   */
  it('问出口的话只能来自追问池,模型没有输出自由文本的通道', async () => {
    const { d } = await ask(session(FULL_ANSWER))
    if (d.followUp) {
      expect(d.source.kind).toBe('pool')
      expect(detail.追问).toContain(d.followUp)
    }
  }, 120_000)

  /**
   * ★ append-only 的收益验证。
   * 要守的性质:**一次请求里只有「新追加的那一段」是未命中的**,前面整场都按 1/30 计费。
   *
   * 两个坑,都踩过:
   * ① 不能断言「命中量单调上升」—— DeepSeek 缓存按 64 token 分块,尾部不足一块不入缓存,
   *    命中量会停在块边界上。
   * ② 不能拿两次请求互相比 —— 前面的用例可能已经把同样的 prompt 缓存过了,数字会被污染。
   *    所以用一次性的 nonce 让这条用例的前缀独一无二。
   */
  it('长会话里,未命中的只是新追加的那一段', async () => {
    const nonce = `本条用例的随机标记 ${Date.now()}。`
    const q = record(targetId, [nonce + FULL_ANSWER])
    const input = { ...session(nonce + FULL_ANSWER), history: [q, q, q] }
    const built = buildSession(input)

    // 第一次:整条前缀都是新的,用来把缓存灌进去
    await chat('live', built.messages, { maxTokens: 200 })
    // 第二次:同样的前缀 + 再追加一题
    const grown = buildSession({ ...input, history: [q, q, q, q] })
    const r = await chat('live', grown.messages, { maxTokens: 200 })

    const uncached = r.promptTokens - r.cachedTokens
    console.log(
      `  4 题共 ${r.promptTokens} tok,未命中 ${uncached}(${Math.round((uncached / r.promptTokens) * 100)}%)`,
    )
    // 未命中的应该只是新追加的一题,占总量的一小部分
    expect(uncached).toBeLessThan(r.promptTokens * 0.4)
    expect(r.cachedTokens).toBeGreaterThan(r.promptTokens * 0.6)
  }, 200_000)
})
