import { describe, it, expect } from 'vitest'
import {
  buildCorpus,
  extractExamPoints,
  loadQuestionDetail,
  loadToneSamples,
  EXCLUDED_CATEGORIES,
} from '../lib/interview/corpus'
import { buildSession, estimateTokens, SOFT_LIMIT } from '../lib/interview/context'
import type { QuestionRecord, SessionInput } from '../lib/interview/context'
import { configFor, extractJson, resolveTurn } from '../lib/interview/llm'
import type { RawTurn } from '../lib/interview/llm'
import { eligible, pick, seededRandom, weightOf, EMPTY_SESSION } from '../lib/interview/sampler'
import type { SamplerProfile, SessionState } from '../lib/interview/sampler'
import { listResumes, loadProfile, samplerProfileFor } from '../lib/interview/resume'
import type { Candidate } from '../lib/interview/types'
import { cacheKey } from '../lib/interview/ttsCache'

const corpus = buildCorpus()
const resumes = listResumes()

function cand(over: Partial<Candidate> & Pick<Candidate, 'id' | 'chapter' | 'article'>): Candidate {
  return {
    kind: 'question',
    ask: over.ask ?? over.id,
    highfreq: false,
    mastered: false,
    fromInterview: false,
    ...over,
  }
}

/** 采样 n 次,返回各章节命中次数 */
function sampleChapters(
  pool: readonly Candidate[],
  profile: SamplerProfile,
  n = 1000,
  session: SessionState = EMPTY_SESSION,
): Record<string, number> {
  const rnd = seededRandom(42)
  const hits: Record<string, number> = {}
  for (let i = 0; i < n; i++) {
    const c = pick(pool, profile, session, rnd)
    if (c) hits[c.chapter] = (hits[c.chapter] ?? 0) + 1
  }
  return hits
}

describe('语料索引 corpus', () => {
  it('题库全部可解析,题目数与磁盘一致', () => {
    expect(corpus.stats.题目总数).toBeGreaterThan(90)
    expect(corpus.questions.every((q) => q.id.includes('/'))).toBe(true)
  })

  it('手撕代码整类排除,且排除后没有匹配不到文章的题', () => {
    expect(EXCLUDED_CATEGORIES).toContain('手撕代码')
    expect(corpus.candidates.some((c) => c.id.startsWith('q:手撕代码/'))).toBe(false)
    // 匹配不到文章的题会静默消失,必须归零——否则是索引漏了
    expect(corpus.stats.匹配不到文章的题目).toBe(0)
  })

  it('占位稿不进出题池(没有正文=没有答案,拿它出题就是编造)', () => {
    const placeholders = new Set(
      corpus.articles.filter((a) => a.state === 'placeholder').map((a) => a.title),
    )
    expect(placeholders.size).toBeGreaterThan(0)
    const leaked = corpus.candidates.filter(
      (c) => c.kind === 'exam-point' && placeholders.has(c.article),
    )
    expect(leaked).toEqual([])
  })

  it('候选池非空且题库题占主力', () => {
    const q = corpus.candidates.filter((c) => c.kind === 'question').length
    expect(q).toBeGreaterThan(80)
    expect(corpus.stats.候选池大小).toBeGreaterThan(q)
  })

  it('考点表两种格式都能解析', () => {
    const table = extractExamPoints(
      ['## 五、面试考点串联', '', '| 高频问法 | 本文哪一节 |', '|---|---|', '| A 是什么? | 二 |', '', '## 相关文献'].join('\n'),
    )
    expect(table).toEqual([{ ask: 'A 是什么?', where: '二' }])

    const list = extractExamPoints(
      ['## 六、面试考点串联', '', '1. B 改了什么 →「动机 + 对比表」', '', '## 相关文献'].join('\n'),
    )
    expect(list).toEqual([{ ask: 'B 改了什么', where: '动机 + 对比表' }])
  })
})

describe('简历 = 硬门禁的来源', () => {
  it('两份简历都能读到,且都声明了在册章节', () => {
    expect(resumes.length).toBe(2)
    for (const r of resumes) {
      expect(r.chapters.length).toBeGreaterThan(0)
      expect(r.name).toBeTruthy()
    }
  })

  it('恰好一份是默认档', () => {
    expect(resumes.filter((r) => r.isDefault).length).toBe(1)
  })

  it('画像里的知识点名**全部**是真实文章名(接地保证)', () => {
    const real = new Set(corpus.articles.map((a) => a.title))
    const bogus: string[] = []
    for (const r of resumes) {
      const p = loadProfile(r.slug)
      if (!p) continue
      for (const name of Object.keys(p.topic亲和)) if (!real.has(name)) bogus.push(`${r.slug}: ${name}`)
      for (const d of p.项目深挖点 ?? []) {
        for (const t of d.关联topic ?? []) if (!real.has(t)) bogus.push(`${r.slug} 深挖点: ${t}`)
      }
    }
    expect(bogus).toEqual([])
  })

  it('画像只给 2/3 分,0 分(永不出题)只能人工加', () => {
    for (const r of resumes) {
      const p = loadProfile(r.slug)
      if (!p) continue
      expect(Object.values(p.topic亲和).every((v) => v === 2 || v === 3)).toBe(true)
    }
  })

  it('画像确实改变了抽题分布', () => {
    const r = resumes.find((x) => x.slug === '02-infra')!
    const p = loadProfile(r.slug)
    expect(p).not.toBeNull()
    const run = (prof: typeof p) => {
      const sp = samplerProfileFor(r, prof)
      const rnd = seededRandom(99)
      const hit: Record<string, number> = {}
      for (let i = 0; i < 3000; i++) {
        const c = pick(corpus.candidates, sp, EMPTY_SESSION, rnd)
        if (c) hit[c.article] = (hit[c.article] ?? 0) + 1
      }
      return hit
    }
    const before = run(null)
    const after = run(p)
    // 简历里明确写过 vLLM,画像该把它抬上去
    expect(after['vLLM'] ?? 0).toBeGreaterThan(before['vLLM'] ?? 0)
  })

  it('没有画像时兜底亲和度为 1,系统照常可用', () => {
    const r = resumes[0]
    const p = samplerProfileFor(r, null)
    expect(p.chapters).toEqual(r.chapters)
    expect(eligible(corpus.candidates, p).length).toBeGreaterThan(0)
  })
})

describe('门禁:章节不在册的永不被抽中', () => {
  it('05-多模态 确实在池子里(否则下面的断言是空跑)', () => {
    const mm = corpus.candidates.filter((c) => c.chapter === '05-多模态')
    expect(mm.length).toBeGreaterThan(0)
  })

  it('两份简历都不含 05-多模态 → 采样 1000 次命中 0', () => {
    for (const r of resumes) {
      expect(r.chapters).not.toContain('05-多模态')
      const hits = sampleChapters(corpus.candidates, samplerProfileFor(r, null))
      expect(hits['05-多模态'] ?? 0).toBe(0)
    }
  })

  it('Agent 档在册的 06-应用 能被抽到', () => {
    const agent = resumes.find((r) => r.chapters.includes('06-应用'))
    expect(agent).toBeDefined()
    const hits = sampleChapters(corpus.candidates, samplerProfileFor(agent!, null))
    expect(hits['06-应用'] ?? 0).toBeGreaterThan(0)
  })

  it('亲和度显式为 0 的文章,即使章节在册也永不被抽中', () => {
    const target = corpus.candidates[0]
    const profile: SamplerProfile = {
      chapters: [target.chapter],
      affinity: { [target.article]: 0 },
      fallbackAffinity: 1,
    }
    expect(eligible(corpus.candidates, profile).some((c) => c.article === target.article)).toBe(false)
    const rnd = seededRandom(7)
    for (let i = 0; i < 1000; i++) {
      expect(pick(corpus.candidates, profile, EMPTY_SESSION, rnd)?.article).not.toBe(target.article)
    }
  })

  it('章节全不在册时返回 null,不退化成随机乱抽', () => {
    const profile: SamplerProfile = { chapters: ['99-不存在'], affinity: {} }
    expect(pick(corpus.candidates, profile)).toBeNull()
  })
})

describe('加权:highfreq 抬高、mastered 压低', () => {
  const pool: Candidate[] = [
    cand({ id: 'a', chapter: 'C', article: 'A' }),
    cand({ id: 'b', chapter: 'C', article: 'B', highfreq: true }),
    cand({ id: 'c', chapter: 'C', article: 'C', mastered: true }),
  ]
  const profile: SamplerProfile = { chapters: ['C'], affinity: {}, fallbackAffinity: 1 }

  it('highfreq 命中率 ≥ 普通题的 1.8 倍', () => {
    const rnd = seededRandom(1)
    const hits: Record<string, number> = {}
    for (let i = 0; i < 20000; i++) {
      const c = pick(pool, profile, EMPTY_SESSION, rnd)!
      hits[c.id] = (hits[c.id] ?? 0) + 1
    }
    expect(hits.b / hits.a).toBeGreaterThanOrEqual(1.8)
  })

  it('mastered 命中率 ≤ 普通题的 0.3 倍', () => {
    const rnd = seededRandom(2)
    const hits: Record<string, number> = {}
    for (let i = 0; i < 20000; i++) {
      const c = pick(pool, profile, EMPTY_SESSION, rnd)!
      hits[c.id] = (hits[c.id] ?? 0) + 1
    }
    expect(hits.c / hits.a).toBeLessThanOrEqual(0.3)
  })

  it('本场已问过的会被压到很低,但不是永久出局', () => {
    const c = pool[0]
    const fresh = weightOf(c, profile)
    const asked = weightOf(c, profile, { asked: new Set(['a']) })
    expect(asked).toBeLessThan(fresh)
    expect(asked).toBeGreaterThan(0)
  })

  it('连着问同一篇会被压低', () => {
    const c = pool[0]
    expect(weightOf(c, profile, { asked: new Set(), lastArticle: 'A' })).toBeLessThan(
      weightOf(c, profile, { asked: new Set(), lastArticle: 'Z' }),
    )
  })
})

describe('上下文:append-only(前缀缓存的全部前提)', () => {
  const tone = loadToneSamples()
  const resume = resumes.find((r) => r.slug === '02-infra')!
  const ids = corpus.candidates.filter((c) => c.kind === 'question').slice(0, 4).map((c) => c.id.slice(2))

  function record(id: string, nTurns: number): QuestionRecord {
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
      turns: Array.from({ length: nTurns }, (_, i) => ({
        ask: i === 0 ? d.题目 : `追问 ${i}`,
        answer: `第 ${i} 轮的回答内容。`.repeat(10),
        verdict: '{"hit":[0],"miss":[1]}',
      })),
    }
  }

  const session = (nQuestions: number, mode: SessionInput['mode'] = 'interview'): SessionInput => ({
    mode,
    toneSamples: tone,
    resumeName: resume.name,
    resumeBody: resume.body,
    history: ids.slice(0, nQuestions - 1).map((id) => record(id, 2)),
    current: record(ids[nQuestions - 1], 1),
  })

  it('消息序列的骨架正确', () => {
    const m = buildSession(session(1)).messages
    expect(m[0].role).toBe('system')
    expect(m[1].role).toBe('user')
    expect(m[1].content).toContain('候选人简历')
    // 简历是**全文**不是摘要 —— 云端装得下,面试官才能深挖项目
    expect(m[1].content).toContain('PrefixGrouper')
    expect(m[m.length - 1].role).toBe('user')
  })

  /**
   * ★ 这条是缓存设计的命根子。
   * DeepSeek 缓存命中价是未命中价的 1/30,前提是**消息一旦写下就永不修改**。
   * 只要有一条历史消息被改动,从那里往后全部按未命中计费,而且没有任何报错提示你。
   */
  it('加轮次、加题目都只在末尾追加,前面的消息一个字节都不变', () => {
    const grow = [
      buildSession(session(1)).messages,
      buildSession(session(2)).messages,
      buildSession(session(3)).messages,
      buildSession(session(4)).messages,
    ]
    for (let i = 1; i < grow.length; i++) {
      const prev = grow[i - 1]
      const next = grow[i]
      expect(next.length).toBeGreaterThan(prev.length)
      // 前一轮的消息(除最后一条,它带着判卷指令)必须原样出现在新一轮的开头
      for (let k = 0; k < prev.length - 1; k++) {
        expect(next[k]).toEqual(prev[k])
      }
    }
  })

  it('可缓存比例随轮次上升', () => {
    const r1 = buildSession(session(1))
    const r4 = buildSession(session(4))
    const pct = (b: typeof r1) => b.cacheableTokens / b.estimatedTokens
    expect(pct(r4)).toBeGreaterThan(pct(r1))
    expect(pct(r4)).toBeGreaterThan(0.8)
  })

  it('参考答案带着「绝不念给他听」的约束一起进上下文', () => {
    const all = buildSession(session(2)).messages.map((m) => m.content).join('\n')
    expect(all).toContain('绝不念给他听')
    expect(all).toContain('绝不把参考答案念出来')
  })

  it('输出格式说明在 system 里(整场不变才能全程命中缓存)', () => {
    const m = buildSession(session(2)).messages
    expect(m[0].content).toContain('只输出一段 JSON')
    // 最后一条必须是纯粹的回答 —— 拼上指令的话,这一轮变成历史时内容会变,前缀就断了
    expect(m.at(-1)!.role).toBe('user')
    expect(m.at(-1)!.content).not.toContain('只输出一段 JSON')
  })

  it('面试档不要评价字段(省一半输出 token),过题档要', () => {
    const iv = buildSession(session(1, 'interview')).messages[0].content
    const dr = buildSession(session(1, 'drill')).messages[0].content
    expect(iv).not.toContain('"评价"')
    expect(dr).toContain('"评价"')
  })

  it('面经原话被当成口气样本注入系统提示', () => {
    expect(tone.length).toBeGreaterThan(0)
    expect(buildSession(session(1)).messages[0].content).toContain(tone[0])
  })

  it('token 估算保守(宁可高估)', () => {
    expect(estimateTokens('中文一二三四五')).toBe(7)
    expect(estimateTokens('abcdefg')).toBe(2)
    expect(estimateTokens('')).toBe(0)
  })

  it('一场 30 轮仍远低于软警戒线', () => {
    const long: SessionInput = {
      ...session(4),
      history: Array.from({ length: 29 }, (_, i) => record(ids[i % ids.length], 2)),
    }
    const built = buildSession(long)
    expect(built.overSoftLimit).toBe(false)
    expect(built.estimatedTokens).toBeLessThan(SOFT_LIMIT)
  })

  it('全部真实题目都能组装,不抛错', () => {
    const bad: string[] = []
    for (const c of corpus.candidates.filter((x) => x.kind === 'question')) {
      try {
        buildSession({ ...session(1), current: record(c.id.slice(2), 1) })
      } catch (e) {
        bad.push(`${c.id}: ${(e as Error).message}`)
      }
    }
    expect(bad).toEqual([])
  })
})

// scripts/interview-index.mjs 是 corpus.ts 的 .mjs 镜像(脚本跑不了 .ts)。
// 这条把两边钉死:改一边忘了改另一边,这里立刻红。
describe('防漂移:统计脚本与 corpus.ts 必须一致', () => {
  it('两边算出的池子规模逐项相同', async () => {
    const script = await import('../scripts/interview-index.mjs')
    expect(script.collect().stats).toEqual(corpus.stats)
  })
})

describe('LLM 输出解析(防幻觉的最后一道闸)', () => {
  const pool = ['追问零', '追问一', '追问二']

  it('从裹了代码块和废话的输出里也能抠出 JSON', () => {
    expect(extractJson<RawTurn>('```json\n{"hit":[0]}\n```')).toEqual({ hit: [0] })
    expect(extractJson<RawTurn>('好的,结果如下:{"hit":[1],"fu":2} 完毕')).toEqual({ hit: [1], fu: 2 })
    expect(() => extractJson('模型今天不想输出 JSON')).toThrow(/没有输出 JSON/)
  })

  it('追问正文由我们查表得到,不信模型复述', () => {
    const d = resolveTurn({ hit: [0], miss: [1], next: 'followup', fu: 1 }, pool, 3)
    expect(d.followUp).toBe('追问一')
    expect(d.source).toEqual({ kind: 'pool', index: 1 })
  })

  it('序号越界 → 退化成换题,绝不凭空冒出文档外的追问', () => {
    const d = resolveTurn({ next: 'followup', fu: 99 }, pool, 3)
    expect(d.next).toBe('nextq')
    expect(d.followUp).toBe('')
  })

  it('池子为空时允许模型自己写,但来源如实标注', () => {
    const d = resolveTurn({ next: 'followup', fu: -1, ask: '那你说说代价是什么?' }, [], 3)
    expect(d.source).toEqual({ kind: 'generated' })
    expect(d.followUp).toBe('那你说说代价是什么?')
  })

  it('越界的要点序号被丢掉,hit 与 miss 不重叠', () => {
    const d = resolveTurn({ hit: [0, 9, -1], miss: [0, 1], next: 'nextq' }, pool, 3)
    expect(d.hit).toEqual([0])
    expect(d.miss).toEqual([1])
  })

  it('分阶段配置:某一阶段能被单独覆盖,不影响其它阶段', () => {
    const base = configFor('live').baseUrl
    expect(base).toBeTruthy()
    process.env.INTERVIEW_LLM_POST_BASE_URL = 'https://api.example.com/v1'
    expect(configFor('post').baseUrl).toBe('https://api.example.com/v1')
    expect(configFor('live').baseUrl).toBe(base) // live 不受影响
    delete process.env.INTERVIEW_LLM_POST_BASE_URL
  })
})

describe('TTS 预渲染缓存', () => {
  it('缺省字段不改变 key —— 否则漏传一个参数缓存就静默全失效', () => {
    const base = { model: 'm', voice: '', instruct: 'i', speed: 1.2, normalize: true }
    const k = cacheKey('题干', base)
    expect(cacheKey('题干', { ...base, voice: undefined })).toBe(k)
    expect(cacheKey('题干', { ...base, normalize: undefined })).toBe(k)
    expect(cacheKey(' 题干 ', base)).toBe(k)
  })

  it('影响音频的参数一变,key 就变', () => {
    const base = { model: 'm', voice: '', instruct: 'i', speed: 1.2, normalize: true }
    const k = cacheKey('题干', base)
    for (const patch of [{ speed: 1.3 }, { instruct: 'j' }, { model: 'n' }, { normalize: false }]) {
      expect(cacheKey('题干', { ...base, ...patch })).not.toBe(k)
    }
  })

  it('防漂移:预渲染脚本与 lib 的 key 实现必须一致', async () => {
    const script = await import('../scripts/interview-prerender.mjs')
    const cfg = { model: 'qwen3-tts-design', voice: '', instruct: '面试官', speed: 1.2, normalize: true }
    expect(script.cacheKey('随便一段题干', cfg)).toBe(cacheKey('随便一段题干', cfg))
  })
})

describe('可复现性', () => {
  it('同种子同结果', () => {
    const r = resumes[0]
    const p = samplerProfileFor(r, null)
    const seq = (seed: number) => {
      const rnd = seededRandom(seed)
      return Array.from({ length: 30 }, () => pick(corpus.candidates, p, EMPTY_SESSION, rnd)?.id)
    }
    expect(seq(123)).toEqual(seq(123))
    expect(seq(123)).not.toEqual(seq(456))
  })

  it('换简历 → 抽题分布明显不同', () => {
    const [a, b] = resumes
    const ha = sampleChapters(corpus.candidates, samplerProfileFor(a, null))
    const hb = sampleChapters(corpus.candidates, samplerProfileFor(b, null))
    expect(Object.keys(ha).sort()).not.toEqual(Object.keys(hb).sort())
  })
})
