import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
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
import { listResumes, loadProfile, parseChapters, samplerProfileFor } from '../lib/interview/resume'
import {
  INTRO_ID,
  PHASE_CAPS,
  PHASE_ORDER,
  introMaterial,
  parsePhaseId,
  projectId,
  projectMaterial,
  projectOrder,
  projectSection,
} from '../lib/interview/phases'
import type { Candidate } from '../lib/interview/types'
import { cacheKey } from '../lib/interview/ttsCache'
import { extractTheme, readSession } from '../lib/interview/session'
import { articleBody, examPointMaterial, sectionOf } from '../lib/interview/kbdrill'
import { materialBlock } from '../lib/interview/prompts'

const corpus = buildCorpus()
const resumes = listResumes()

function cand(over: Partial<Candidate> & Pick<Candidate, 'id' | 'chapter' | 'article'>): Candidate {
  return {
    kind: 'question',
    ask: over.ask ?? over.id,
    highfreq: false,
    mastered: false,
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

  it('题库语料全部是真题,不再保留面经来源分支', () => {
    expect(corpus.questions.every((q) => q.tags.includes('真题'))).toBe(true)
    expect(corpus.questions.every((q) => !q.tags.includes('面经'))).toBe(true)
  })

  it('面试官口气样本直接来自题库,不依赖另一套原题档案', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'iprep-tone-'))
    try {
      fs.mkdirSync(path.join(root, 'RAG'))
      fs.writeFileSync(path.join(root, 'RAG', '001.md'), [
        '---',
        'topic: RAG/基础',
        'summary: RAG 为什么需要重排',
        'tags: [真题]',
        'mastered: false',
        '---',
        '',
        '## 题目',
        '',
        'RAG 为什么需要重排,不用向量相似度直接返回?',
        '',
        '## 答案',
        '',
        '因为召回与精排目标不同。',
      ].join('\n'))
      expect(loadToneSamples(2, root)).toEqual([
        'RAG 为什么需要重排,不用向量相似度直接返回?',
      ])
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('手撕代码整类排除,未关联文章的普通题必须已登记待写', () => {
    expect(EXCLUDED_CATEGORIES).toContain('手撕代码')
    expect(corpus.candidates.some((c) => c.id.startsWith('q:手撕代码/'))).toBe(false)
    const unmatched = corpus.questions.filter(
      (q) => !EXCLUDED_CATEGORIES.includes(q.category) && !q.chapter,
    )
    const articleMap = fs.readFileSync(
      new URL('../docs/04-知识库地图.md', import.meta.url), 'utf8',
    )
    for (const q of unmatched) {
      // 不放过意外拼错的 topic;未成文 topic 必须已经登记为计划文章。
      expect(articleMap, q.id).toContain(`\`${q.article}\``)
      expect(corpus.candidates.some((c) => c.id === `q:${q.id}`)).toBe(false)
    }
    expect(corpus.stats.匹配不到文章的题目).toBe(unmatched.length)
  })

  it('新增 SFT 的 LoRA 真题进入预训练与微调章', () => {
    const id = 'q:SFT/002-lora-原理与秩选择.md'
    expect(corpus.candidates.find((c) => c.id === id)).toMatchObject({
      kind: 'question', chapter: '02-预训练与微调', article: 'LoRA',
    })
    const pool = eligible(corpus.candidates, { chapters: ['02-预训练与微调'], affinity: {} })
    expect(pool.some((c) => c.id === id)).toBe(true)
    expect(pool.every((c) => c.chapter === '02-预训练与微调')).toBe(true)
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
    // ⚠️ 必须比**章内占比**,不能比绝对次数。改成两级采样之后,
    // 一个章节被抽中的总次数由简历的章节权重决定,和画像无关 ——
    // 拿绝对次数比会把「Infra 整体份额下降」误当成「画像没生效」。
    const shareInChapter = (prof: typeof p, article: string, chapter: string) => {
      const sp = samplerProfileFor(r, prof)
      const rnd = seededRandom(99)
      let inChapter = 0
      let hit = 0
      for (let i = 0; i < 6000; i++) {
        const c = pick(corpus.candidates, sp, EMPTY_SESSION, rnd)
        if (!c) continue
        if (c.chapter === chapter) inChapter++
        if (c.article === article) hit++
      }
      return inChapter ? hit / inChapter : 0
    }
    // 简历里明确写过 vLLM,画像该把它在本章内的占比抬上去
    expect(shareInChapter(p, 'vLLM', '04-Infra')).toBeGreaterThan(
      shareInChapter(null, 'vLLM', '04-Infra'),
    )
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

  it('历史来源字段不能再改变真题权重', () => {
    const base = cand({ id: 'a', chapter: 'C', article: 'A' })
    const legacy = { ...base, fromInterview: true } as Candidate
    const profile: SamplerProfile = { chapters: ['C'], affinity: {} }
    expect(weightOf(legacy, profile)).toBe(weightOf(base, profile))
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

  it('题库真题原话被当成口气样本注入系统提示', () => {
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
  }, 15_000)
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

describe('复盘主题:模型不配合时必须有兜底', () => {
  it('正常给了主题 → 摘出来,并从正文剥掉', () => {
    const { theme, body } = extractTheme('主题:CUDA Graph 与 MoE 并行\n\n## 整体\n\n还行。')
    expect(theme).toBe('CUDA Graph 与 MoE 并行')
    expect(body.startsWith('## 整体')).toBe(true)
    expect(body).not.toContain('主题')
  })

  it('全角冒号 / 加粗 / 前导空白都要认', () => {
    expect(extractTheme('  **主题**:投机解码\n\n## 整体').theme).toBe('投机解码')
    expect(extractTheme('主题: KV 量化\n正文').theme).toBe('KV 量化')
  })

  // 这条是重点:复盘是自由 markdown,开不了 JSON 模式,模型漏写迟早发生。
  // 漏写时绝不能让整场记录写不进去 —— 返回 null,由 writeSession 回落到「覆盖」。
  it('模型没写主题 → 返回 null 且正文一字不动', () => {
    const raw = '## 整体\n\n他基础还行。\n\n## 下一步\n\n- 看 CUDA Graph'
    const { theme, body } = extractTheme(raw)
    expect(theme).toBeNull()
    expect(body).toBe(raw)
  })

  it('只写了「主题:」后面空着 → 也当没写(别把下一行的标题当主题)', () => {
    expect(extractTheme('主题:\n\n## 整体').theme).toBeNull()
    expect(extractTheme('主题:   \n## 整体').theme).toBeNull()
  })

  it('模型写成一大段 → 截断,别撑爆列表标题', () => {
    const long = '主题:' + '很长'.repeat(40) + '\n\n## 整体'
    expect(extractTheme(long).theme!.length).toBeLessThanOrEqual(40)
  })
})

describe('会话文件:没有 frontmatter 的旧稿也要读得出来', () => {
  it('第一场真实面试(无 frontmatter)能解析出全部字段', () => {
    const found = readSession('2026-09-02-2337-agent-rl')
    if (!found) return // 本地没这个文件就跳过,不让测试依赖某一场记录
    const s = found.summary
    expect(s.date).toBe('2026-09-02')
    expect(s.time).toBe('23:37')
    expect(s.简历).toBe('Agent 档')
    expect(s.模式).toBe('过题档')
    expect(s.轮次).toBe(5)
    expect(s.hit).toBe(4)
    expect(s.points).toBe(20)
    expect(s.覆盖.length).toBeGreaterThan(0)
    // 没有 frontmatter 就没有主题 → 必须回落到覆盖的前两篇,不能是空标题
    expect(s.主题).not.toBe('')
    expect(s.主题).toBe(s.覆盖.slice(0, 2).join(' · '))
  })

  it('读不存在的场次 → null,不抛异常', () => {
    expect(readSession('2099-01-01-0000-nope')).toBeNull()
  })
})

describe('两级采样:章节大小不再决定出题比例', () => {
  const r = resumes.find((x) => x.slug === '01-agent-rl')!
  const dist = (weights: Record<string, number>) => {
    const sp = { ...samplerProfileFor(r, loadProfile(r.slug)), chapterWeights: weights }
    const rnd = seededRandom(7)
    const by: Record<string, number> = {}
    let n = 0
    for (let i = 0; i < 3000; i++) {
      const c = pick(corpus.candidates, sp, EMPTY_SESSION, rnd)
      if (c) { by[c.chapter] = (by[c.chapter] ?? 0) + 1; n++ }
    }
    return Object.fromEntries(Object.entries(by).map(([k, v]) => [k, v / n]))
  }

  // 这条是这次改动的全部理由:改之前 Agent 档和 Infra 档抽出来几乎一样
  // (Infra 73% vs 75%),因为候选池 55% 是 Infra,简历那一层等于没生效。
  it('章节权重高的方向,出题占比就高', () => {
    const d = dist({ '03-强化学习': 4, '06-应用': 3, '02-预训练与微调': 2, '04-Infra': 2, '01-模型结构': 1 })
    expect(d['03-强化学习']).toBeGreaterThan(d['04-Infra'])
    expect(d['06-应用']).toBeGreaterThan(d['01-模型结构'])
  })

  it('把 Infra 权重拉满 → 它才占多数(说明是权重在起作用,不是语料量)', () => {
    const d = dist({ '03-强化学习': 1, '04-Infra': 8 })
    expect(d['04-Infra']).toBeGreaterThan(0.6)
  })

  it('等权时,语料多 6 倍的 Infra 也不该压过强化学习太多', () => {
    const d = dist({ '03-强化学习': 1, '04-Infra': 1 })
    // 一级采样下这个比值会是 6 倍以上;两级采样应该接近 1
    expect(d['04-Infra'] / d['03-强化学习']).toBeLessThan(2)
  })

  it('权重 0 或负数 = 不在册,一道都不出', () => {
    const { chapters, chapterWeights } = parseChapters({ '04-Infra': 3, '03-强化学习': 0 })
    expect(chapters).toEqual(['04-Infra'])
    expect(chapterWeights['03-强化学习']).toBeUndefined()
  })

  it('老的列表写法照样认,各章等权', () => {
    const { chapters, chapterWeights } = parseChapters(['04-Infra', '03-强化学习'])
    expect(chapters).toEqual(['04-Infra', '03-强化学习'])
    expect(chapterWeights).toEqual({ '04-Infra': 1, '03-强化学习': 1 })
  })
})

describe('面试档阶段机', () => {
  const r = resumes.find((x) => x.slug === '01-agent-rl')!
  const p = loadProfile(r.slug)

  it('阶段顺序固定,且每个阶段都有代码卡的轮次上限', () => {
    expect(PHASE_ORDER).toEqual(['intro', 'project', 'tech', 'breadth'])
    expect(PHASE_CAPS.intro).toBeLessThanOrEqual(3) // 自我介绍不该耗太多轮
    expect(PHASE_CAPS.project).toBeGreaterThan(PHASE_CAPS.intro)
    expect(PHASE_CAPS.breadth).toBe(Infinity) // 广度阶段由模型决定何时收
  })

  it('自我介绍不判分:要点为空,所以 hit/miss 必然都是空数组', () => {
    const m = introMaterial(p)
    expect(m.要点).toEqual([])
    expect(resolveTurn({ hit: [0, 1], miss: [2] }, [], 0)).toMatchObject({ hit: [], miss: [] })
  })

  it('项目深挖的要点 = 五个固定维度,判卷内核不用改', () => {
    const m = projectMaterial(r, p, 0)
    expect(m).not.toBeNull()
    expect(m!.要点).toHaveLength(5)
    expect(m!.要点.join()).toMatch(/背景/)
    expect(m!.要点.join()).toMatch(/权衡/)
  })

  // 防幻觉:项目内容只能来自简历原文,抠不到宁可回落也不编
  it('参考答案是简历原文里那个项目的段落', () => {
    const m = projectMaterial(r, p, 0)!
    const name = p!.项目深挖点[0].项目
    expect(m.答案).toContain(name)
    expect(r.body).toContain(m.答案.split('\n')[0].replace(/^#+\s*/, ''))
  })

  it('简历里没有这个项目 → 返回空段落,绝不编造', () => {
    expect(projectSection(r.body, '完全不存在的项目')).toBe('')
  })

  it('项目问完了 → projectMaterial 返回 null,由前端顺次进下一阶段', () => {
    expect(projectMaterial(r, p, 999)).toBeNull()
  })

  it('阶段合成 id 能来回解析,且不会和题库 id 撞', () => {
    expect(parsePhaseId(INTRO_ID)).toEqual({ kind: 'intro' })
    expect(parsePhaseId(projectId(2))).toEqual({ kind: 'project', index: 2 })
    expect(parsePhaseId('AI Infra/071-cudagraph-收益与判据.md')).toBeNull()
  })

  it('模型的 nextphase 信号能被解析出来', () => {
    expect(resolveTurn({ next: 'nextphase' }, ['追问一'], 3).next).toBe('nextphase')
  })
})

// 这条盯的是「不报错但行为不对」:技术延伸阶段接不上项目时,
// 早期版本会悄悄退回全池,于是页面标着「技术延伸」却在问毫无关系的题。
describe('技术延伸:接不上就跳过,不许假装接上了', () => {
  it('没有关联题时空池不能抽题,即使完整题池还有其他题', () => {
    // 固定语料,避免新增题目后「某项目暂时没有关联题」这一偶然前提失效。
    const pool = [cand({ id: 'unrelated', chapter: '04-Infra', article: '解码策略' })]
    const topics = new Set(['GRPO'])
    const profile: SamplerProfile = { chapters: ['04-Infra', '03-强化学习'], affinity: {} }
    const matched = pool.filter((c) => c.kind === 'question' && topics.has(c.article))
    expect(pick(pool, profile, EMPTY_SESSION, seededRandom(1))?.id).toBe('unrelated')
    expect(pick(matched, profile, EMPTY_SESSION, seededRandom(1))).toBeNull()
  })
})

describe('项目挑选:交给模型,但随机性由代码注入', () => {
  const p = loadProfile('01-agent-rl')!
  const n = p.项目深挖点.length

  // 这是这次改动的核心洞察:判卷调用是 temperature 0,模型是确定性的。
  // 只把「挑哪个项目」交给模型,同一份简历每次会选同一个 —— 交给 LLM ≠ 随机。
  // 所以呈现顺序必须由代码按会话种子打乱。
  it('不同会话种子 → 项目呈现顺序不同', () => {
    const orders = new Set([1, 2, 3, 4, 5, 6, 7, 8].map((s) => projectOrder(n, s).join(',')))
    expect(orders.size).toBeGreaterThan(1)
  })

  it('同一个种子 → 顺序完全一致(整场稳定,材料块不会变)', () => {
    expect(projectOrder(n, 12345)).toEqual(projectOrder(n, 12345))
  })

  it('打乱只是换顺序,不增不减不重复', () => {
    const o = projectOrder(n, 999)
    expect([...o].sort()).toEqual(Array.from({ length: n }, (_, i) => i))
  })

  it('开场材料里带着项目清单,模型才知道能挑什么', () => {
    const m = introMaterial(p, 42)
    expect(m.extra).toBeTruthy()
    expect(m.extra).toContain('proj:')
    for (const d of p.项目深挖点) expect(m.extra).toContain(d.项目)
  })

  it('没有画像时开场也能用,只是没有项目清单', () => {
    const m = introMaterial(null, 42)
    expect(m.要点).toEqual([])
    expect(m.extra).toBeUndefined()
  })

  it('模型的 proj 会被解析出来,负数与非数字一律忽略', () => {
    expect(resolveTurn({ next: 'nextphase', proj: 2 }, [], 0).project).toBe(2)
    expect(resolveTurn({ next: 'nextphase', proj: -1 }, [], 0).project).toBeUndefined()
    expect(resolveTurn({ next: 'nextphase' }, [], 0).project).toBeUndefined()
  })
})

describe('阶段 id 不能被当成题库 id 切', () => {
  // 题库 id 是 `q:<分类>/<文件>`,要剥前两个字符;阶段 id 是 `phase:project:0`,一个字都不能剥。
  // 写成 `.slice(2)` 会得到 `ase:project:0` —— 服务端查不到材料、复盘的 history 为空,
  // 表现是**整场面试结束后没有复盘,而且不报错**。实测踩到,所以钉一条。
  // **白名单,不是排除法**:排除法每加一种新 id 就再坏一次(已栽两次)
  it('只有 q: 剥前缀,phase: 与 k: 原样保留', () => {
    const toQuestionId = (id: string) => (id.startsWith('q:') ? id.slice(2) : id)
    expect(toQuestionId('q:AI Infra/071-cudagraph-收益与判据.md')).toBe(
      'AI Infra/071-cudagraph-收益与判据.md',
    )
    expect(toQuestionId('phase:intro')).toBe('phase:intro')
    expect(toQuestionId('phase:project:0')).toBe('phase:project:0')
    expect(toQuestionId('k:CudaGraph#9')).toBe('k:CudaGraph#9')
  })

  it('被切坏的阶段 id 解析不出来 —— 这正是复盘丢失的机制', () => {
    expect(parsePhaseId('ase:project:0')).toBeNull()
    expect(parsePhaseId('phase:project:0')).toEqual({ kind: 'project', index: 0 })
  })
})

describe('追问不许重复问', () => {
  const pool = ['你说的那个 2.3 倍是怎么测出来的?']

  it('模型再挑同一条已问过的追问 → 不重复,退化成换题', () => {
    const first = resolveTurn({ next: 'followup', fu: 0 }, pool, 3, ['主问'])
    expect(first.next).toBe('followup')
    expect(first.followUp).toBe(pool[0])
    // 同一条已经问过了
    const again = resolveTurn({ next: 'followup', fu: 0 }, pool, 3, ['主问', pool[0]])
    expect(again.next).toBe('nextq')
  })

  it('自己写的追问重复了也挡住', () => {
    const r = resolveTurn({ next: 'followup', fu: -1, ask: '那个数字怎么测的?' }, [], 3, [
      '主问',
      '那个数字怎么测的?',
    ])
    expect(r.next).toBe('nextq')
  })

  it('换了个说法就放行', () => {
    const r = resolveTurn({ next: 'followup', fu: -1, ask: '换个规模还成立吗?' }, [], 3, ['主问'])
    expect(r.next).toBe('followup')
    expect(r.source).toEqual({ kind: 'generated' })
  })
})

describe('单篇过题', () => {
  const body = articleBody('CudaGraph')!

  it('考点表的「本文哪一节」能定位到原文小节', () => {
    const a = corpus.articles.find((x) => x.title === 'CudaGraph')!
    const located = a.examPoints.filter((p) => sectionOf(body, p.where))
    expect(located.length).toBe(a.examPoints.length)
  })

  it('四种 where 写法都认(旧稿各写各的)', () => {
    const fake = ['## 一、甲', 'A', '## 二、乙', 'B', '## 三、丙丁', 'C'].join('\n')
    expect(sectionOf(fake, '二(压缩过的答案)')).toContain('乙')
    expect(sectionOf(fake, '第三节')).toContain('丙丁')
    expect(sectionOf(fake, '一段很长的解释(§二)')).toContain('乙')
    expect(sectionOf(fake, '3.2')).toContain('丙丁')
    // 只写小节标题关键词(旧稿的写法)。**关键词至少两个字** ——
    // 一个字太容易在别的标题里误命中,宁可定位不到回落整篇
    expect(sectionOf(fake, '丙丁')).toContain('丙丁')
    expect(sectionOf(fake, '丙')).toBe('')
  })

  // 参考答案错了比没有更糟 —— 它会直接把判分带偏
  it('定位不到就返回空串,由调用方回落整篇,绝不猜一节', () => {
    const fake = ['## 一、甲', 'A'].join('\n')
    expect(sectionOf(fake, '完全对不上的东西')).toBe('')
    expect(sectionOf(fake, '九')).toBe('')
  })

  it('考点材料:要点留空但标了 selfPoints,参考答案是原文', () => {
    const m = examPointMaterial('CudaGraph', 0)!
    expect(m.要点).toEqual([])
    expect(m.selfPoints).toBe(true)
    expect(m.答案.length).toBeGreaterThan(100)
    expect(m.article).toBe('CudaGraph')
  })

  it('materialBlock 要能区分「现抽清单」和「不判分」', () => {
    const ep = materialBlock({ ...examPointMaterial('CudaGraph', 0)! })
    expect(ep).toContain('没有现成的要点清单')
    expect(ep).not.toContain('不判分')
    const intro = materialBlock({ ...introMaterial(loadProfile('01-agent-rl')) })
    expect(intro).toContain('不判分')
  })

  // 没有人写的清单时,inRange 必须以模型现抽的那份为基准,
  // 否则 pointCount=0 会把所有 hit/miss 过滤光 —— 判分静默变成 0/0
  it('模型现抽的要点当基准,hit/miss 不会被过滤光', () => {
    const r = resolveTurn(
      { hit: [0, 2], miss: [1], 要点: ['甲', '乙', '丙'] },
      [],
      0, // 没有人写的清单
    )
    expect(r.points).toEqual(['甲', '乙', '丙'])
    expect(r.hit).toEqual([0, 2])
    expect(r.miss).toEqual([1])
  })

  it('有人写的清单时,模型的要点被忽略(以人写的为准)', () => {
    const r = resolveTurn({ hit: [0], 要点: ['模型瞎写的'] }, [], 3)
    expect(r.points).toEqual([])
    expect(r.hit).toEqual([0])
  })

  // buildCorpus 里有去重:有题库题的文章不再加它的考点行。
  // 单篇过题要的正好相反 —— 所以池子必须自己组,不能直接筛 candidates。
  it('CudaGraph 有题库题,所以它的考点行不在通用候选池里', () => {
    const inPool = corpus.candidates.filter(
      (c) => c.kind === 'exam-point' && c.article === 'CudaGraph',
    )
    expect(inPool).toEqual([])
    expect(corpus.articles.find((a) => a.title === 'CudaGraph')!.examPoints.length).toBeGreaterThan(10)
  })

  it('单篇上下文换成文章全文,不是简历', () => {
    const rec = {
      material: { 题目: 'q', 要点: [], 答案: 'a', 追问: [], needsReview: false, examPoints: [] },
      turns: [{ ask: 'q', answer: 'x' }],
    }
    const built = buildSession({
      mode: 'drill', toneSamples: [], resumeName: 'CudaGraph', resumeBody: body,
      sourceKind: 'article', history: [], current: rec,
    })
    expect(built.messages[1].content).toContain('本次考察的文章(CudaGraph)')
    expect(built.messages[1].content).not.toContain('候选人简历')
  })
})

describe('「考一遍这篇」的显示条件', () => {
  const qByArticle: Record<string, number> = {}
  for (const c of corpus.candidates) {
    if (c.kind === 'question') qByArticle[c.article] = (qByArticle[c.article] ?? 0) + 1
  }
  // 页面上的判据,和 app/kb/[...slug]/page.tsx 保持一致
  const drillCount = (a: (typeof corpus.articles)[number]) =>
    a.state === 'placeholder' ? 0 : a.examPoints.length + (qByArticle[a.title] ?? 0)

  it('成文与旧稿只要有考点表或题库题就能考', () => {
    const written = corpus.articles.filter((a) => a.state !== 'placeholder')
    expect(written.every((a) => drillCount(a) > 0)).toBe(true)
    expect(written.length).toBeGreaterThan(60)
  })

  // 仓库已有的规则:没有正文=没有答案,拿它出题就是编造。
  // 有 6 篇占位稿恰好有题库题指过来,不挡的话按钮会出现在一篇还没写的文章上,
  // 而送进模型的「本次考察的文章」只是段 200 字的占位提示。
  it('占位稿一律不能考,哪怕它有题库题', () => {
    const holes = corpus.articles.filter((a) => a.state === 'placeholder')
    expect(holes.some((a) => (qByArticle[a.title] ?? 0) > 0)).toBe(true) // 确实存在这种情况
    expect(holes.every((a) => drillCount(a) === 0)).toBe(true)
  })
})
