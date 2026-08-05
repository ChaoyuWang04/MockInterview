# 本地刷题系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec(`docs/superpowers/specs/2026-08-05-interview-prep-design.md`)实现本地、单人、markdown 文件即数据库的面试刷题系统,一条 `npm run dev` 可用。

**Architecture:** Next.js 15 App Router 全栈单体。服务端组件直接调 `lib/questions.ts` 读文件;唯一写接口 `PATCH /api/question` 做白名单校验后定点写回 md 文件(frontmatter `mastered` 行 / `## Note` 分区),原子替换。渲染统一走 `components/Markdown.tsx`(GFM + KaTeX + 代码高亮 + Mermaid 客户端渲染)。

**Tech Stack:** Node 25(本机已装,项目隔离靠 node_modules)、Next.js 15 + React 19 + TypeScript、Tailwind CSS v4 + @tailwindcss/typography、gray-matter、react-markdown + remark-gfm/remark-math/rehype-katex/rehype-highlight、mermaid、Vitest。

**约定(贯穿全部任务):**
- 顶层 `## ` 只保留给六个分区名:`题目/要点/答案/知识点/追问/Note`;围栏代码块内的 `## ` 不算分区标题
- frontmatter 仅 `mastered` 必需(缺省按 false);`difficulty/tags/company` 可选
- 写回 = 读取 → 定点字符串修改 → 写 `.tmp` → rename;绝不用 gray-matter 重新序列化 frontmatter
- 以 `_` 或 `.` 开头的文件/文件夹不算题目(如 `_template.md`)

---

### Task 1: 脚手架与依赖

**Files:**
- Create: Next.js 项目骨架(create-next-app 生成)
- Create: `.claude/launch.json`

- [ ] **Step 1: 脚手架**(目录已含 `docs/` 与 `.git`,create-next-app 的空目录检查允许这两者)

```bash
cd /Users/samwong/Desktop/1Project/interviewprep
npx --yes create-next-app@15 . --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-npm
```

若报目录非空:在 `/tmp/ip-scaffold` 脚手架后 `rsync -a --exclude node_modules /tmp/ip-scaffold/ ./`,再 `npm install`。

- [ ] **Step 2: 安装运行依赖与测试依赖**

```bash
npm i gray-matter react-markdown remark-gfm remark-math rehype-katex katex rehype-highlight mermaid @tailwindcss/typography
npm i -D vitest
```

- [ ] **Step 3: 加 test 脚本** — package.json `scripts` 增加 `"test": "vitest run"`

- [ ] **Step 4: 创建 `vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: { environment: 'node', include: ['tests/**/*.test.ts'] },
})
```

- [ ] **Step 5: 创建 `.claude/launch.json`**

```json
{
  "version": "0.0.1",
  "configurations": [
    { "name": "dev", "runtimeExecutable": "npm", "runtimeArgs": ["run", "dev"], "port": 3000 }
  ]
}
```

- [ ] **Step 6: 验证 dev server 能启动**(启动、curl 200、停止)
- [ ] **Step 7: Commit** `chore: Next.js 15 脚手架与依赖`

---

### Task 2: 核心模块 lib/questions.ts(TDD)

**Files:**
- Create: `lib/types.ts`
- Create: `lib/questions.ts`
- Test: `tests/questions.test.ts`

- [ ] **Step 1: 写 `lib/types.ts`**

```ts
export const SECTION_NAMES = ['题目', '要点', '答案', '知识点', '追问', 'Note'] as const
export type SectionName = (typeof SECTION_NAMES)[number]

export interface QuestionMeta {
  difficulty?: string
  tags: string[]
  company?: string
  mastered: boolean
}

export interface Question {
  category: string
  file: string
  meta: QuestionMeta
  sections: Partial<Record<SectionName, string>>
  error?: string
}

export interface CategoryStat {
  name: string
  total: number
  mastered: number
}

export interface Stats {
  total: number
  mastered: number
  categories: CategoryStat[]
}
```

- [ ] **Step 2: 写失败测试 `tests/questions.test.ts`**(全文如下)

```ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  splitSections, loadQuestion, loadCategory, listCategories,
  getStats, isValidRef, setMastered, saveNote,
} from '../lib/questions'

let root: string

const SAMPLE = `---
difficulty: 简单        # 简单 | 中等 | 困难
tags: [RAG, 查询扩展]
company: 字节
mastered: false         # 程序写回
---

## 题目

什么是 HyDE?

## 答案

先生成假设文档再检索。

\`\`\`python
## 这行井号在代码块里,不是分区标题
print("hi")
\`\`\`

### 补充

内部小标题用三级。

## Note

旧笔记
`

const SAMPLE_NOTE_MIDDLE = `---
mastered: true
---

## 题目

Q

## 答案

A

## Note

旧笔记

## 追问

- 追问一条
`

function write(category: string, file: string, content: string) {
  fs.mkdirSync(path.join(root, category), { recursive: true })
  fs.writeFileSync(path.join(root, category, file), content, 'utf8')
}

function read(category: string, file: string) {
  return fs.readFileSync(path.join(root, category, file), 'utf8')
}

beforeEach(() => { root = fs.mkdtempSync(path.join(os.tmpdir(), 'iprep-')) })
afterEach(() => { fs.rmSync(root, { recursive: true, force: true }) })

describe('splitSections', () => {
  it('按保留 ## 分区名切分,代码块内的 ## 不误切', () => {
    const { content } = splitForTest(SAMPLE)
    const s = splitSections(content)
    expect(Object.keys(s).sort()).toEqual(['Note', '答案', '题目'].sort())
    expect(s['题目']).toBe('什么是 HyDE?')
    expect(s['答案']).toContain('## 这行井号在代码块里')
    expect(s['答案']).toContain('### 补充')
    expect(s['Note']).toBe('旧笔记')
  })
})

// 帮助函数:去掉 frontmatter,拿到正文(与 loadQuestion 内部一致)
function splitForTest(raw: string): { content: string } {
  const m = raw.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/)
  return { content: m ? raw.slice(m[0].length) : raw }
}

describe('loadQuestion', () => {
  it('解析 frontmatter 与分区', () => {
    write('RAG', '001-hyde.md', SAMPLE)
    const q = loadQuestion('RAG', '001-hyde.md', root)
    expect(q.error).toBeUndefined()
    expect(q.meta).toEqual({ difficulty: '简单', tags: ['RAG', '查询扩展'], company: '字节', mastered: false })
  })

  it('缺少必填分区报错但不抛异常', () => {
    write('RAG', 'bad.md', '---\nmastered: false\n---\n\n## 题目\n\n只有题目\n')
    const q = loadQuestion('RAG', 'bad.md', root)
    expect(q.error).toContain('答案')
  })

  it('frontmatter 缺省字段有默认值', () => {
    write('RAG', 'min.md', '## 题目\n\nQ\n\n## 答案\n\nA\n')
    const q = loadQuestion('RAG', 'min.md', root)
    expect(q.meta.mastered).toBe(false)
    expect(q.meta.tags).toEqual([])
    expect(q.error).toBeUndefined()
  })
})

describe('setMastered', () => {
  it('只改 mastered 行,保留行内注释与其余全部字节', () => {
    write('RAG', '001.md', SAMPLE)
    setMastered('RAG', '001.md', true, root)
    expect(read('RAG', '001.md')).toBe(
      SAMPLE.replace('mastered: false         # 程序写回', 'mastered: true         # 程序写回'),
    )
  })

  it('frontmatter 无 mastered 键时追加', () => {
    write('RAG', 'nokey.md', '---\ndifficulty: 中等\n---\n\n## 题目\n\nQ\n\n## 答案\n\nA\n')
    setMastered('RAG', 'nokey.md', true, root)
    const raw = read('RAG', 'nokey.md')
    expect(raw).toContain('difficulty: 中等\nmastered: true\n---')
  })

  it('无 frontmatter 时补一个', () => {
    write('RAG', 'nofm.md', '## 题目\n\nQ\n\n## 答案\n\nA\n')
    setMastered('RAG', 'nofm.md', true, root)
    expect(read('RAG', 'nofm.md').startsWith('---\nmastered: true\n---\n')).toBe(true)
  })
})

describe('saveNote', () => {
  it('只替换 Note 分区,前文逐字节不变', () => {
    write('RAG', '001.md', SAMPLE)
    saveNote('RAG', '001.md', '新笔记内容', root)
    const raw = read('RAG', '001.md')
    const prefix = SAMPLE.slice(0, SAMPLE.indexOf('## Note') + '## Note'.length)
    expect(raw.startsWith(prefix)).toBe(true)
    expect(raw).toContain('新笔记内容')
    expect(raw).not.toContain('旧笔记')
  })

  it('Note 在中间时,后续分区逐字节保留', () => {
    write('RAG', 'mid.md', SAMPLE_NOTE_MIDDLE)
    saveNote('RAG', 'mid.md', '改过的笔记', root)
    const raw = read('RAG', 'mid.md')
    expect(raw).toContain('改过的笔记')
    expect(raw).not.toContain('旧笔记')
    expect(raw).toContain('## 追问\n\n- 追问一条\n')
  })

  it('保存空内容 = 清空正文保留标题', () => {
    write('RAG', '001.md', SAMPLE)
    saveNote('RAG', '001.md', '', root)
    const raw = read('RAG', '001.md')
    expect(raw).toContain('## Note')
    expect(raw).not.toContain('旧笔记')
  })

  it('无 Note 分区时追加到末尾', () => {
    write('RAG', 'nonote.md', '## 题目\n\nQ\n\n## 答案\n\nA\n')
    saveNote('RAG', 'nonote.md', '追加的笔记', root)
    expect(read('RAG', 'nonote.md')).toContain('## Note\n\n追加的笔记\n')
  })

  it('写回后重新解析,Note 内容一致(往返)', () => {
    write('RAG', '001.md', SAMPLE)
    saveNote('RAG', '001.md', '带**加粗**和 $x^2$ 的笔记', root)
    const q = loadQuestion('RAG', '001.md', root)
    expect(q.sections['Note']).toBe('带**加粗**和 $x^2$ 的笔记')
  })
})

describe('目录扫描与校验', () => {
  it('列出分类与题目,忽略 _ 开头', () => {
    write('RAG', '001.md', SAMPLE)
    write('RAG', '_template.md', '模板')
    write('Agent', '001.md', SAMPLE)
    fs.mkdirSync(path.join(root, '_drafts'), { recursive: true })
    expect(listCategories(root)).toEqual(['Agent', 'RAG'])
    expect(loadCategory('RAG', root).map(q => q.file)).toEqual(['001.md'])
  })

  it('isValidRef 白名单校验,拒绝路径穿越', () => {
    write('RAG', '001.md', SAMPLE)
    expect(isValidRef('RAG', '001.md', root)).toBe(true)
    expect(isValidRef('RAG', '002.md', root)).toBe(false)
    expect(isValidRef('..', '001.md', root)).toBe(false)
    expect(isValidRef('RAG', '../001.md', root)).toBe(false)
  })

  it('getStats 统计,解析错误的题不计入已掌握', () => {
    write('RAG', '001.md', SAMPLE)
    write('RAG', '002.md', SAMPLE_NOTE_MIDDLE)
    write('RAG', 'broken.md', '---\nmastered: true\n---\n\n没有任何分区\n')
    const s = getStats(root)
    expect(s.total).toBe(3)
    expect(s.mastered).toBe(1)
    expect(s.categories).toEqual([{ name: 'RAG', total: 3, mastered: 1 }])
  })
})
```

- [ ] **Step 3: 跑测试确认失败** — `npm test`,预期:模块不存在而失败
- [ ] **Step 4: 实现 `lib/questions.ts`**(全文如下)

```ts
import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'
import { SECTION_NAMES, type SectionName, type Question, type Stats } from './types'

export function questionsRoot(): string {
  return path.join(process.cwd(), 'questions')
}

function isVisible(name: string): boolean {
  return !name.startsWith('.') && !name.startsWith('_')
}

export function listCategories(root = questionsRoot()): string[] {
  if (!fs.existsSync(root)) return []
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory() && isVisible(d.name))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

export function listQuestionFiles(category: string, root = questionsRoot()): string[] {
  const dir = path.join(root, category)
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isFile() && d.name.endsWith('.md') && isVisible(d.name))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 白名单校验:category/file 必须真实存在于扫描结果中,天然杜绝路径穿越 */
export function isValidRef(category: string, file: string, root = questionsRoot()): boolean {
  return listCategories(root).includes(category) && listQuestionFiles(category, root).includes(file)
}

interface FenceState {
  char: string
  len: number
}

function fenceOpen(line: string): FenceState | null {
  const m = line.match(/^\s*(`{3,}|~{3,})/)
  return m ? { char: m[1][0], len: m[1].length } : null
}

function fenceClose(line: string, fence: FenceState): boolean {
  const m = line.match(/^\s*(`{3,}|~{3,})\s*$/)
  return !!m && m[1][0] === fence.char && m[1].length >= fence.len
}

function reservedHeading(line: string): SectionName | null {
  const m = line.match(/^##\s+(.+?)\s*$/)
  if (m && (SECTION_NAMES as readonly string[]).includes(m[1])) return m[1] as SectionName
  return null
}

/** 按保留的 ## 分区标题切分正文(围栏代码块内的 ## 不算) */
export function splitSections(body: string): Partial<Record<SectionName, string>> {
  const acc: Partial<Record<SectionName, string[]>> = {}
  let fence: FenceState | null = null
  let current: SectionName | null = null
  for (const line of body.split('\n')) {
    if (fence) {
      if (fenceClose(line, fence)) fence = null
      if (current) acc[current]!.push(line)
      continue
    }
    const open = fenceOpen(line)
    if (open) {
      fence = open
      if (current) acc[current]!.push(line)
      continue
    }
    const heading = reservedHeading(line)
    if (heading) {
      current = heading
      acc[current] = []
      continue
    }
    if (current) acc[current]!.push(line)
  }
  const out: Partial<Record<SectionName, string>> = {}
  for (const name of SECTION_NAMES) {
    const lines = acc[name]
    if (lines !== undefined) out[name] = lines.join('\n').trim()
  }
  return out
}

export function loadQuestion(category: string, file: string, root = questionsRoot()): Question {
  const base: Question = { category, file, meta: { tags: [], mastered: false }, sections: {} }
  let raw: string
  try {
    raw = fs.readFileSync(path.join(root, category, file), 'utf8')
  } catch (e) {
    return { ...base, error: `读取文件失败:${(e as Error).message}` }
  }
  let data: Record<string, unknown>
  let content: string
  try {
    const parsed = matter(raw)
    data = parsed.data
    content = parsed.content
  } catch (e) {
    return { ...base, error: `frontmatter 解析失败:${(e as Error).message}` }
  }
  const q: Question = {
    category,
    file,
    meta: {
      difficulty: typeof data.difficulty === 'string' ? data.difficulty : undefined,
      tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
      company: typeof data.company === 'string' ? data.company : undefined,
      mastered: data.mastered === true,
    },
    sections: splitSections(content),
  }
  if (!q.sections['题目']) q.error = '缺少必填分区「## 题目」'
  else if (!q.sections['答案']) q.error = '缺少必填分区「## 答案」'
  return q
}

export function loadCategory(category: string, root = questionsRoot()): Question[] {
  return listQuestionFiles(category, root).map((f) => loadQuestion(category, f, root))
}

export function getStats(root = questionsRoot()): Stats {
  const categories = listCategories(root).map((name) => {
    const qs = loadCategory(name, root)
    return { name, total: qs.length, mastered: qs.filter((q) => !q.error && q.meta.mastered).length }
  })
  return {
    total: categories.reduce((s, c) => s + c.total, 0),
    mastered: categories.reduce((s, c) => s + c.mastered, 0),
    categories,
  }
}

function atomicWrite(filePath: string, content: string): void {
  const tmp = `${filePath}.${process.pid}.tmp`
  fs.writeFileSync(tmp, content, 'utf8')
  fs.renameSync(tmp, filePath)
}

const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---/
// 三组:键与空白 / 现值 / 行尾(空白+注释),定点替换保注释
const MASTERED_LINE_RE = /^(mastered:[ \t]*)([^\s#]*)([ \t]*(?:#[^\n]*)?)$/m

/** 定点改写 frontmatter 的 mastered 行,绝不整体重新序列化 */
export function setMastered(category: string, file: string, mastered: boolean, root = questionsRoot()): void {
  const filePath = path.join(root, category, file)
  const raw = fs.readFileSync(filePath, 'utf8')
  const value = String(mastered)
  const m = raw.match(FRONTMATTER_RE)
  let next: string
  if (m) {
    const fm = m[1]
    const updated = MASTERED_LINE_RE.test(fm)
      ? fm.replace(MASTERED_LINE_RE, `$1${value}$3`)
      : `${fm}\nmastered: ${value}`
    next = `---\n${updated}\n---` + raw.slice(m[0].length)
  } else {
    next = `---\nmastered: ${value}\n---\n\n${raw}`
  }
  atomicWrite(filePath, next)
}

/** 替换 ## Note 分区正文;无该分区则追加;空内容清空正文保留标题 */
export function saveNote(category: string, file: string, note: string, root = questionsRoot()): void {
  const filePath = path.join(root, category, file)
  const raw = fs.readFileSync(filePath, 'utf8')
  const trimmed = note.trim()
  const lines = raw.split('\n')

  // 跳过 frontmatter
  let i = 0
  if (lines[0]?.trim() === '---') {
    i = 1
    while (i < lines.length && lines[i].trim() !== '---') i++
    i++
  }

  // 围栏感知扫描,定位 ## Note 标题行与分区结束(下一个保留 ## 或 EOF)
  let fence: FenceState | null = null
  let noteStart = -1
  let noteEnd = lines.length
  for (; i < lines.length; i++) {
    const line = lines[i]
    if (fence) {
      if (fenceClose(line, fence)) fence = null
      continue
    }
    const open = fenceOpen(line)
    if (open) {
      fence = open
      continue
    }
    const heading = reservedHeading(line)
    if (!heading) continue
    if (noteStart >= 0) {
      noteEnd = i
      break
    }
    if (heading === 'Note') noteStart = i
  }

  let next: string
  if (noteStart >= 0) {
    const before = lines.slice(0, noteStart + 1).join('\n')
    const after = lines.slice(noteEnd).join('\n')
    const body = trimmed ? `\n\n${trimmed}\n` : '\n'
    next = after ? `${before}${body}\n${after}` : `${before}${body}`
  } else {
    if (!trimmed) return // 没有分区且内容为空:无事可做
    const base = raw.endsWith('\n') ? raw : `${raw}\n`
    next = `${base}\n## Note\n\n${trimmed}\n`
  }
  atomicWrite(filePath, next)
}
```

- [ ] **Step 5: 跑测试确认全绿** — `npm test`
- [ ] **Step 6: Commit** `feat: 核心模块——题库扫描/解析/定点写回`

---

### Task 3: 示例题库与模板

**Files:**
- Create: `questions/_template.md`
- Create: `questions/RAG/001-检索优化技术.md`(含表格)
- Create: `questions/RAG/002-embedding-模型选型.md`(mastered: true,验证统计)
- Create: `questions/Agent/001-react-模式.md`(含 mermaid)
- Create: `questions/手撕代码/001-手写-self-attention.md`(含 KaTeX + python 代码)

- [ ] **Step 1: 创建 `questions/_template.md`**

````markdown
---
difficulty: 中等        # 简单 | 中等 | 困难
tags: []                # 知识点标签,如 [RAG, 重排序]
company:                # 可选,出题公司
mastered: false         # 程序写回,新题保持 false
---

## 题目

(必填)题干。支持表格、$行内公式$、$$块级公式$$、```mermaid 图、代码块。
注意:正文内小标题请用 ### 及以下,## 只保留给分区名。

## 要点

- (可选)答出这道题应覆盖的要点

## 答案

(必填)参考答案。

## 知识点

(可选)核心知识点总结。

## 追问

- (可选)面试官可能的追问

## Note
````

- [ ] **Step 2: 创建四道示例题**(内容按上述文件清单,分别覆盖表格 / mermaid / KaTeX+代码;`RAG/002` 设 `mastered: true`。执行时按 `_template.md` 结构撰写原创内容,不照抄老师平台文本)
- [ ] **Step 3: 验证解析** — 临时脚本或 `npx tsx` 调 `getStats()`,预期 total=4, mastered=1,无 error
- [ ] **Step 4: Commit** `feat: 题目模板与示例题库`

---

### Task 4: PATCH API

**Files:**
- Create: `app/api/question/route.ts`

- [ ] **Step 1: 实现 route**(全文如下)

```ts
import { NextResponse } from 'next/server'
import { isValidRef, saveNote, setMastered } from '@/lib/questions'

export async function PATCH(req: Request) {
  let body: unknown
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: '请求体不是合法 JSON' }, { status: 400 })
  }
  const { category, file, mastered, note } = (body ?? {}) as Record<string, unknown>

  if (typeof category !== 'string' || typeof file !== 'string')
    return NextResponse.json({ error: 'category/file 必填' }, { status: 400 })
  if (mastered === undefined && note === undefined)
    return NextResponse.json({ error: 'mastered/note 至少提供一项' }, { status: 400 })
  if (mastered !== undefined && typeof mastered !== 'boolean')
    return NextResponse.json({ error: 'mastered 必须是布尔值' }, { status: 400 })
  if (note !== undefined && typeof note !== 'string')
    return NextResponse.json({ error: 'note 必须是字符串' }, { status: 400 })
  if (!isValidRef(category, file))
    return NextResponse.json({ error: '题目不存在' }, { status: 400 })

  try {
    if (mastered !== undefined) setMastered(category, file, mastered)
    if (note !== undefined) saveNote(category, file, note)
  } catch (e) {
    return NextResponse.json({ error: `写入失败:${(e as Error).message}` }, { status: 500 })
  }
  return NextResponse.json({ ok: true })
}
```

- [ ] **Step 2: 启动 dev,curl 验证**
  - 合法请求 mastered=true → `{"ok":true}`;`git diff questions/` 仅 mastered 一行变化
  - 再置回 false → git diff 干净
  - 非法(`file: "../x.md"`、缺参数、非布尔)→ 400
- [ ] **Step 3: Commit** `feat: PATCH /api/question 写回接口`

---

### Task 5: Markdown 渲染管线

**Files:**
- Create: `components/Markdown.tsx`
- Create: `components/Mermaid.tsx`
- Modify: `app/layout.tsx`(样式导入、语言、metadata)
- Modify: `app/globals.css`(网格背景、typography 插件、mermaid 容器样式)

- [ ] **Step 1: `components/Mermaid.tsx`**

```tsx
'use client'

import { useEffect, useId, useState } from 'react'

export default function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/[^a-zA-Z0-9]/g, '')
  const [svg, setSvg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const mermaid = (await import('mermaid')).default
        mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'strict' })
        const { svg } = await mermaid.render(`mmd${id}`, chart)
        if (!cancelled) setSvg(svg)
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => { cancelled = true }
  }, [chart, id])

  if (error)
    return (
      <div className="mermaid-container border border-red-200 bg-red-50 p-3 text-sm">
        <pre className="overflow-x-auto whitespace-pre-wrap">{chart}</pre>
        <p className="mt-2 text-red-600">Mermaid 渲染失败:{error}</p>
      </div>
    )
  if (!svg) return <div className="mermaid-container text-sm text-gray-400">图表渲染中…</div>
  return <div className="mermaid-container" dangerouslySetInnerHTML={{ __html: svg }} />
}
```

- [ ] **Step 2: `components/Markdown.tsx`**

```tsx
'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import Mermaid from './Mermaid'

export default function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex, [rehypeHighlight, { plainText: ['mermaid'] }]]}
      components={{
        code({ className, children: code, ...props }) {
          if (className?.includes('language-mermaid')) return <Mermaid chart={String(code ?? '')} />
          return (
            <code className={className} {...props}>
              {code}
            </code>
          )
        },
      }}
    >
      {children}
    </ReactMarkdown>
  )
}
```

- [ ] **Step 3: 改 `app/layout.tsx`**(去掉模板字体,引 katex/hljs 样式)

```tsx
import type { Metadata } from 'next'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github.css'
import './globals.css'

export const metadata: Metadata = {
  title: '刷题系统',
  description: '本地大模型面试刷题系统',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen text-gray-900 antialiased">{children}</body>
    </html>
  )
}
```

- [ ] **Step 4: 改 `app/globals.css`**(覆盖模板内容)

```css
@import 'tailwindcss'
@plugin '@tailwindcss/typography';

body {
  background-color: #fafafa;
  background-image:
    linear-gradient(rgba(0, 0, 0, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 0, 0, 0.03) 1px, transparent 1px);
  background-size: 56px 56px;
}

/* mermaid 容器脱离 pre 的代码样式 */
.prose pre:has(.mermaid-container) {
  background: transparent;
  padding: 0;
}
.mermaid-container svg {
  max-width: 100%;
  height: auto;
}
```

(注意第一行 `@import 'tailwindcss';` 要带分号——执行时以正确语法为准)

- [ ] **Step 5: `npx tsc --noEmit` 通过;Commit** `feat: markdown 渲染管线(GFM/KaTeX/高亮/Mermaid)`

---

### Task 6: 主页

**Files:**
- Modify: `app/page.tsx`(覆盖模板)
- Create: `components/CategoryCard.tsx`

- [ ] **Step 1: `components/CategoryCard.tsx`**

```tsx
import Link from 'next/link'
import type { CategoryStat } from '@/lib/types'

export default function CategoryCard({ stat }: { stat: CategoryStat }) {
  const pct = stat.total === 0 ? 0 : Math.round((stat.mastered / stat.total) * 100)
  return (
    <Link
      href={`/${encodeURIComponent(stat.name)}`}
      className="block border border-gray-200 bg-white p-5 transition-colors hover:border-gray-400"
    >
      <div className="flex items-baseline justify-between">
        <span className="font-semibold">{stat.name}</span>
        <span className="font-mono text-sm text-gray-400">{stat.total} 题</span>
      </div>
      <div className="mt-4 h-1.5 w-full bg-gray-100">
        <div className="h-full bg-blue-600" style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-2 flex justify-between text-xs">
        <span className="text-green-700">{stat.mastered} 已掌握</span>
        <span className="font-mono text-gray-400">{pct}%</span>
      </div>
    </Link>
  )
}
```

- [ ] **Step 2: `app/page.tsx`**

```tsx
import CategoryCard from '@/components/CategoryCard'
import { getStats } from '@/lib/questions'

export const dynamic = 'force-dynamic'

export default function Home() {
  const stats = getStats()
  return (
    <main className="mx-auto max-w-5xl px-6 py-16">
      <h1 className="text-2xl font-bold">大模型面试刷题</h1>
      <div className="mt-10">
        <div className="font-mono text-6xl font-bold text-green-700">
          {stats.mastered} <span className="text-gray-300">/</span> {stats.total}
        </div>
        <p className="mt-2 text-sm text-gray-500">已掌握 / 总题数</p>
      </div>
      <div className="mt-16 mb-6 flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">题库分类</h2>
        <span className="font-mono text-xs text-gray-400">
          {stats.categories.length} 个分类 · {stats.total} 题
        </span>
      </div>
      {stats.categories.length === 0 ? (
        <p className="text-gray-500">
          questions/ 下还没有题目。复制 questions/_template.md 到新建的分类文件夹开始出题,规范见 docs/question-authoring.md。
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {stats.categories.map((c) => (
            <CategoryCard key={c.name} stat={c} />
          ))}
        </div>
      )}
    </main>
  )
}
```

- [ ] **Step 3: 浏览器验证**(统计数字 4 题 1 掌握、三张分类卡、进度条)
- [ ] **Step 4: Commit** `feat: 主页——总览统计与分类卡片`

---

### Task 7: 刷题页

**Files:**
- Create: `app/[category]/page.tsx`
- Create: `components/QuestionView.tsx`
- Create: `components/NoteEditor.tsx`

- [ ] **Step 1: `app/[category]/page.tsx`**(Next 15 中 params 是 Promise;中文分类名要 decodeURIComponent)

```tsx
import { notFound } from 'next/navigation'
import QuestionView from '@/components/QuestionView'
import { listCategories, loadCategory } from '@/lib/questions'

export const dynamic = 'force-dynamic'

export default async function CategoryPage({ params }: { params: Promise<{ category: string }> }) {
  const { category: raw } = await params
  const category = decodeURIComponent(raw)
  if (!listCategories().includes(category)) notFound()
  return <QuestionView category={category} initialQuestions={loadCategory(category)} />
}
```

- [ ] **Step 2: `components/NoteEditor.tsx`**

```tsx
'use client'

import { useState } from 'react'
import Markdown from './Markdown'

interface Props {
  note: string
  onSave: (note: string) => Promise<boolean>
}

export default function NoteEditor({ note, onSave }: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(note)
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    const ok = await onSave(draft)
    setSaving(false)
    if (ok) setEditing(false)
  }

  return (
    <section className="mt-8 border-t border-dashed border-gray-200 pt-6">
      <div className="flex items-center justify-between">
        <h3 className="font-mono text-xs tracking-widest text-gray-400">NOTE</h3>
        {!editing && (
          <button
            onClick={() => {
              setDraft(note)
              setEditing(true)
            }}
            className="text-sm text-blue-600 hover:underline"
          >
            编辑
          </button>
        )}
      </div>
      {editing ? (
        <div className="mt-3">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={6}
            autoFocus
            placeholder="支持 markdown 语法"
            className="w-full border border-gray-300 bg-white p-3 font-mono text-sm focus:border-blue-500 focus:outline-none"
          />
          <div className="mt-2 flex gap-3">
            <button
              onClick={save}
              disabled={saving}
              className="bg-gray-900 px-4 py-1.5 text-sm text-white disabled:opacity-50"
            >
              {saving ? '保存中…' : '保存'}
            </button>
            <button onClick={() => setEditing(false)} className="text-sm text-gray-500 hover:text-gray-900">
              取消
            </button>
          </div>
        </div>
      ) : note.trim() ? (
        <div className="prose prose-sm mt-3 max-w-none">
          <Markdown>{note}</Markdown>
        </div>
      ) : (
        <p className="mt-3 text-sm text-gray-300">…</p>
      )}
    </section>
  )
}
```

- [ ] **Step 3: `components/QuestionView.tsx`**(核心交互:空格展开、←/→ 切题、标记掌握、Note 保存;textarea/input 聚焦时快捷键失效;切题时收起答案;NoteEditor 用 key 重置)

```tsx
'use client'

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import type { Question, SectionName } from '@/lib/types'
import Markdown from './Markdown'
import NoteEditor from './NoteEditor'

const ANSWER_SECTIONS: { name: SectionName; label: string }[] = [
  { name: '要点', label: 'KEY POINTS' },
  { name: '答案', label: 'REFERENCE ANSWER' },
  { name: '知识点', label: 'KNOWLEDGE POINT' },
  { name: '追问', label: '追问题目' },
]

const DIFFICULTY_STYLE: Record<string, string> = {
  简单: 'text-green-600',
  中等: 'text-amber-600',
  困难: 'text-red-600',
}

interface Props {
  category: string
  initialQuestions: Question[]
}

export default function QuestionView({ category, initialQuestions }: Props) {
  const [questions, setQuestions] = useState(initialQuestions)
  const [index, setIndex] = useState(0)
  const [expanded, setExpanded] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const total = questions.length
  const masteredCount = questions.filter((q) => q.meta.mastered).length
  const q = questions[index]

  const go = useCallback(
    (delta: number) => {
      setIndex((i) => Math.min(total - 1, Math.max(0, i + delta)))
      setExpanded(false)
    },
    [total],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.isContentEditable)) return
      if (e.code === 'Space') {
        e.preventDefault()
        setExpanded((v) => !v)
      } else if (e.key === 'ArrowRight') go(1)
      else if (e.key === 'ArrowLeft') go(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go])

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 2500)
  }

  const patch = async (payload: { mastered?: boolean; note?: string }) => {
    try {
      const res = await fetch('/api/question', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category, file: q.file, ...payload }),
      })
      return res.ok
    } catch {
      return false
    }
  }

  const toggleMastered = async () => {
    const next = !q.meta.mastered
    if (await patch({ mastered: next })) {
      setQuestions((qs) =>
        qs.map((item, i) => (i === index ? { ...item, meta: { ...item.meta, mastered: next } } : item)),
      )
    } else showToast('保存失败,请重试')
  }

  const saveNote = async (note: string) => {
    const ok = await patch({ note })
    if (ok)
      setQuestions((qs) =>
        qs.map((item, i) => (i === index ? { ...item, sections: { ...item.sections, Note: note } } : item)),
      )
    else showToast('Note 保存失败,请重试')
    return ok
  }

  if (total === 0)
    return (
      <main className="mx-auto max-w-4xl px-6 py-16">
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">← 返回主页</Link>
        <p className="mt-10 text-gray-500">
          这个分类还没有题目。复制 questions/_template.md 到 questions/{category}/ 开始出题。
        </p>
      </main>
    )

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <div className="mb-8 flex items-start justify-between">
        <Link href="/" className="text-sm text-gray-500 hover:text-gray-900">← 返回主页</Link>
        <div className="border border-green-700 px-4 py-2 text-right">
          <div className="font-mono text-xs tracking-widest text-gray-400">{category} 掌握</div>
          <div className="font-mono text-lg text-green-700">
            {masteredCount} / {total}
          </div>
          <div className="mt-1 h-1 w-32 bg-gray-100">
            <div
              className="h-full bg-green-600"
              style={{ width: `${total ? (masteredCount / total) * 100 : 0}%` }}
            />
          </div>
        </div>
      </div>

      <article className="border border-gray-200 bg-white p-8">
        {q.error ? (
          <div className="border border-red-300 bg-red-50 p-4 text-sm text-red-700">
            <p className="font-mono">{category}/{q.file}</p>
            <p className="mt-1">{q.error}</p>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3 font-mono text-xs tracking-widest">
              <span className="text-blue-600">{category}</span>
              {q.meta.difficulty && (
                <span className={DIFFICULTY_STYLE[q.meta.difficulty] ?? 'text-gray-500'}>{q.meta.difficulty}</span>
              )}
              <span className="text-gray-400">
                #{index + 1} / {total}
              </span>
            </div>

            <div className="prose mt-4 max-w-none font-semibold">
              <Markdown>{q.sections['题目'] ?? ''}</Markdown>
            </div>

            {(q.meta.tags.length > 0 || q.meta.company) && (
              <div className="mt-4 flex flex-wrap gap-2">
                {q.meta.tags.map((t) => (
                  <span key={t} className="border border-gray-200 px-2 py-0.5 text-xs text-gray-600">{t}</span>
                ))}
                {q.meta.company && (
                  <span className="border border-gray-100 bg-gray-50 px-2 py-0.5 text-xs text-gray-400">{q.meta.company}</span>
                )}
              </div>
            )}

            <div className="mt-8 text-center">
              <button onClick={() => setExpanded((v) => !v)} className="text-blue-600 hover:underline">
                {expanded ? '收起答案' : '展开答案'}
              </button>
              <p className="mt-1 font-mono text-[10px] tracking-widest text-gray-400">SPACE 快捷键</p>
            </div>

            {expanded &&
              ANSWER_SECTIONS.map(({ name, label }) =>
                q.sections[name] ? (
                  <section key={name} className="mt-8 border-t border-dashed border-gray-200 pt-6">
                    <h3 className="mb-3 font-mono text-xs tracking-widest text-gray-400">{label}</h3>
                    <div className="prose prose-sm max-w-none">
                      <Markdown>{q.sections[name]!}</Markdown>
                    </div>
                  </section>
                ) : null,
              )}

            <div className="mt-10 border-t border-gray-200 pt-6">
              <button
                onClick={toggleMastered}
                className={
                  q.meta.mastered
                    ? 'border border-green-700 bg-green-700 px-4 py-2 text-sm text-white'
                    : 'border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:border-green-700 hover:text-green-700'
                }
              >
                {q.meta.mastered ? '✓ 已掌握' : '标记已掌握'}
              </button>
            </div>

            <NoteEditor key={`${category}/${q.file}`} note={q.sections['Note'] ?? ''} onSave={saveNote} />
          </>
        )}
      </article>

      <div className="mt-6 flex items-center justify-between">
        <button
          onClick={() => go(-1)}
          disabled={index === 0}
          className="border border-gray-300 bg-white px-4 py-2 text-sm disabled:opacity-40"
        >
          ← 上一题
        </button>
        <span className="font-mono text-xs text-gray-400">←/→ 切题 · 空格 展开</span>
        <button
          onClick={() => go(1)}
          disabled={index === total - 1}
          className="border border-gray-300 bg-white px-4 py-2 text-sm disabled:opacity-40"
        >
          下一题 →
        </button>
      </div>

      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-gray-900 px-4 py-2 text-sm text-white">
          {toast}
        </div>
      )}
    </main>
  )
}
```

- [ ] **Step 4: `npx tsc --noEmit` 通过**
- [ ] **Step 5: Commit** `feat: 刷题页——展开答案/快捷键/标记掌握/Note`

---

### Task 8: 端到端验收与文档

**Files:**
- Create: `README.md`(覆盖脚手架生成的)
- Create: `docs/question-authoring.md`
- Create: `docs/maintenance.md`

- [ ] **Step 1: 浏览器逐项走验收清单**(spec 第 10 节)
  1. 主页统计与分类卡正确
  2. 进入分类 → 展开/收起答案(按钮 + 空格)、←/→ 切题
  3. 标记已掌握 → 主页与右上角进度联动;`git diff` 仅 mastered 行变化;再取消 → diff 干净
  4. Note 编辑保存 → 文件 `## Note` 分区正确写入;清空保存 → 正文清空标题保留
  5. 表格 / KaTeX / mermaid / 代码高亮四种内容全部正确渲染
  6. `npm test` 全绿;`npm run build` 成功
- [ ] **Step 2: 发现的问题当场修复并重验**
- [ ] **Step 3: 写三份文档**
  - `README.md`:项目是什么、启动(`npm run dev`)、加题三步、目录结构、技术栈
  - `docs/question-authoring.md`:frontmatter 字段表、六分区约定(## 保留、内部用 ###)、表格/公式/mermaid/代码示例、`_template.md` 用法
  - `docs/maintenance.md`:git 习惯(题库即数据,常提交)、备份、依赖升级(`npm outdated`)、常见问题(端口占用、解析错误卡片怎么读、新分类不显示检查 `_`/`.` 前缀)
- [ ] **Step 4: 最终 Commit** `docs: README 与使用维护文档` + 确认 `git status` 干净
