// 给每份简历生成画像:npm run interview:profile
//
// 这是整套系统里**唯一**让 LLM 参与「问什么」决策的地方,而且只给软权重:
//   · 在册章节(硬门禁)从简历 frontmatter 抄,不给模型改的机会
//   · 亲和度 0(永不出题)也不让模型定,只有你手改 JSON 才会出现
// 模型能做的只是「把某些文章从 1 分抬到 2 或 3 分」——抬错了最多是问偏一点,
// 不会问出你完全没碰过的东西。
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import matter from 'gray-matter'
import { collect } from './interview-index.mjs'

const RESUMES = path.join(process.cwd(), 'interview', 'resumes')
const PROFILES = path.join(process.cwd(), 'interview', 'profiles')

const env = (k) => process.env[`INTERVIEW_LLM_PREP_${k}`] ?? process.env[`INTERVIEW_LLM_${k}`]
const BASE_URL = env('BASE_URL') ?? 'http://localhost:1234/v1'
const MODEL = env('MODEL') ?? 'qwen3-14b-mlx'
const API_KEY = env('API_KEY') ?? 'lm-studio'

async function ask(system, user) {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${API_KEY}` },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: 'system', content: system },
        // 画像是「读简历、对着清单做匹配」的分类活,不是推理活。
        // 开着思考会把 token 预算烧光(实测正文直接为空),而且结果反而更差。
        { role: 'user', content: user + ' /no_think' },
      ],
      max_tokens: 3000,
      temperature: 0,
    }),
  })
  if (!res.ok) {
    throw new Error(`LLM ${res.status}: ${(await res.text()).slice(0, 300)}`)
  }
  const d = await res.json()
  const u = d.usage ?? {}
  const think = u.completion_tokens_details?.reasoning_tokens ?? 0
  const content = d.choices?.[0]?.message?.content ?? ''
  if (!content.trim()) {
    throw new Error(
      `模型返回空正文(思考 ${think} tok / 上限 ${u.completion_tokens ?? '?'})。` +
        `多半是思考把预算烧光了,或 finish_reason=${d.choices?.[0]?.finish_reason}`,
    )
  }
  return content
}

function extractJson(raw) {
  const s = raw.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim()
  const a = s.indexOf('{')
  const b = s.lastIndexOf('}')
  if (a < 0 || b <= a) throw new Error(`模型没输出 JSON:${raw.slice(0, 200)}`)
  return JSON.parse(s.slice(a, b + 1))
}

const SYSTEM = `你在为一位候选人做面试准备画像。任务:给定编号的知识点列表,判断
哪些是他简历里**明确做过**的(A 类),哪些是**强相关**的(B 类)。

规则:
1. **只输出编号,不要输出名字** —— 名字由程序查表得到
2. 只依据简历原文判断,不要脑补他"应该也会"什么
3. **宁可多列不要少列**:列进来只是让这个知识点更容易被问到,不会屏蔽任何东西;
   漏列则会让他简历里的强项考不到。一个项目涉及的相邻知识点都应该列上
4. 只输出 JSON,不要代码块,不要解释`

function userPrompt(resume, indexed) {
  const byChapter = {}
  for (const { i, a } of indexed) {
    ;(byChapter[a.chapter] ??= []).push(`${i}=${a.title}`)
  }
  const vocab = Object.entries(byChapter)
    .map(([ch, names]) => `【${ch}】\n${names.join('  ')}`)
    .join('\n')

  return `## 候选人简历

${resume.body}

## 知识点清单(编号=名字)

${vocab}

## 输出(A/B 都填编号,不要填名字)

{
  "A": [简历里明确做过的编号],
  "B": [强相关但简历没直接写的编号],
  "项目深挖点": [
    {"项目":"简历里的项目名","可问":"面试官会怎么往下挖的一个具体问题","编号":[相关知识点编号]}
  ],
  "摘要": "150 字以内,概括他做过什么,给面试官当背景用"
}`
}

async function buildProfile(file) {
  const raw = fs.readFileSync(path.join(RESUMES, file), 'utf8')
  const parsed = matter(raw)
  const resume = {
    slug: file.slice(0, -3),
    name: parsed.data.name ?? file.slice(0, -3),
    chapters: Array.isArray(parsed.data['章节']) ? parsed.data['章节'].map(String) : [],
    body: parsed.content.trim(),
  }
  const hash = 'sha256:' + crypto.createHash('sha256').update(raw).digest('hex').slice(0, 16)
  const out = path.join(PROFILES, `${resume.slug}.json`)

  if (fs.existsSync(out)) {
    try {
      if (JSON.parse(fs.readFileSync(out, 'utf8')).resumeHash === hash) {
        console.log(`  ⏭  ${resume.name.padEnd(10)} 简历没变,跳过`)
        return
      }
    } catch {
      // 画像文件坏了就重生成
    }
  }

  const { articles } = collect()
  // 编号只在本次调用内有效:模型给编号,名字由我们查表 —— 编不出不存在的知识点
  const indexed = articles
    .filter((a) => resume.chapters.includes(a.chapter) && a.usableAsSource)
    .map((a, i) => ({ i, a }))
  process.stdout.write(`  ⏳ ${resume.name.padEnd(10)} 候选知识点 ${indexed.length} 个,思考中…`)

  const t0 = Date.now()
  const got = extractJson(await ask(SYSTEM, userPrompt(resume, indexed)))

  const nameOf = (n) => (Number.isInteger(n) && n >= 0 && n < indexed.length ? indexed[n].a.title : null)
  const affinity = {}
  let bad = 0
  for (const [key, score] of [['B', 2], ['A', 3]]) {
    for (const n of got[key] ?? []) {
      const name = nameOf(n)
      if (name) affinity[name] = score
      else bad++
    }
  }

  const digs = (got['项目深挖点'] ?? []).slice(0, 8).map((d) => ({
    项目: String(d['项目'] ?? ''),
    可问: String(d['可问'] ?? ''),
    // 关联 topic 同样查表,不收模型写的名字
    关联topic: (d['编号'] ?? []).map(nameOf).filter(Boolean),
  }))

  fs.mkdirSync(PROFILES, { recursive: true })
  fs.writeFileSync(
    out,
    JSON.stringify(
      {
        resume: resume.slug,
        resumeHash: hash,
        // 亲和度只有 2 和 3;没列到的按 1(岗位常识,照样会考),
        // 0 = 永不出题,只允许你手改这个文件加进来
        topic亲和: affinity,
        项目深挖点: digs,
        摘要: String(got['摘要'] ?? '').slice(0, 300),
      },
      null,
      2,
    ) + '\n',
    'utf8',
  )

  const n3 = Object.values(affinity).filter((v) => v === 3).length
  const n2 = Object.values(affinity).filter((v) => v === 2).length
  const pct = Math.round(((n3 + n2) / indexed.length) * 100)
  console.log(
    `\r  ✅ ${resume.name.padEnd(10)} 熟悉 ${n3} · 相关 ${n2} · 覆盖 ${pct}% · ${((Date.now() - t0) / 1000).toFixed(1)}s`,
  )
  if (bad) console.log(`     ⚠️  ${bad} 个越界编号被丢弃`)
}

async function main() {
  if (!fs.existsSync(RESUMES)) {
    console.error(`找不到 ${RESUMES}`)
    process.exit(1)
  }
  const files = fs.readdirSync(RESUMES).filter((f) => f.endsWith('.md') && !f.startsWith('_'))
  console.log(`\n简历画像 · ${BASE_URL} · ${MODEL}\n`)
  for (const f of files) {
    try {
      await buildProfile(f)
    } catch (e) {
      console.error(`\r  ❌ ${f}:${e.message}`)
      process.exitCode = 1
    }
  }
  console.log(`\n画像写在 interview/profiles/,可以手改:`)
  console.log(`  · 想让某个知识点永不出现 → 加一行 "知识点名": 0`)
  console.log(`  · 想让它常出现 → 改成 3\n`)
}

main()
