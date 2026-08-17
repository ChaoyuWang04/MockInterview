import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  splitSections, loadQuestion, loadCategory, listCategories,
  getStats, isValidRef, setMastered, setHighFreq, saveNote,
} from '../lib/questions'
import { groupByTopic, sortQuestions } from '../lib/sorting'

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

// 帮助函数:去掉 frontmatter,拿到正文(与 loadQuestion 内部一致)
function splitForTest(raw: string): { content: string } {
  const m = raw.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/)
  return { content: m ? raw.slice(m[0].length) : raw }
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

describe('loadQuestion', () => {
  it('解析 frontmatter 与分区', () => {
    write('RAG', '001-hyde.md', SAMPLE)
    const q = loadQuestion('RAG', '001-hyde.md', root)
    expect(q.error).toBeUndefined()
    expect(q.meta).toEqual({
      difficulty: '简单',
      tags: ['RAG', '查询扩展'],
      company: '字节',
      mastered: false,
      highfreq: false,
      topic: undefined,
      summary: undefined,
    })
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

describe('setHighFreq', () => {
  it('定点写入 highfreq,不动其他任何字节', () => {
    write('RAG', '001.md', SAMPLE)
    setHighFreq('RAG', '001.md', true, root)
    // SAMPLE 里没有 highfreq 键 → 追加到 frontmatter 末尾,正文一字不改
    const raw = read('RAG', '001.md')
    expect(raw).toContain('mastered: false         # 程序写回\nhighfreq: true\n---')
    expect(raw).toContain('## 题目\n\n什么是 HyDE?')
    expect(loadQuestion('RAG', '001.md', root).meta.highfreq).toBe(true)
  })

  it('已有 highfreq 键时定点替换值并保留行内注释', () => {
    write('RAG', 'hf.md', '---\nhighfreq: false   # 高频标记\nmastered: false\n---\n\n## 题目\n\nQ\n\n## 答案\n\nA\n')
    setHighFreq('RAG', 'hf.md', true, root)
    expect(read('RAG', 'hf.md')).toContain('highfreq: true   # 高频标记')
  })

  it('与 mastered 互不干扰', () => {
    write('RAG', '001.md', SAMPLE)
    setHighFreq('RAG', '001.md', true, root)
    setMastered('RAG', '001.md', true, root)
    const meta = loadQuestion('RAG', '001.md', root).meta
    expect([meta.highfreq, meta.mastered]).toEqual([true, true])
  })
})

describe('题库列表排序与分组', () => {
  const q = (file: string, difficulty: string, highfreq: boolean, topic: string) =>
    ({ category: 'RL', file, meta: { tags: [], mastered: false, highfreq, difficulty, topic }, sections: {} })
  const list = [
    q('a.md', '中等', false, 'GRPO/优势'),
    q('b.md', '困难', true, 'GRPO/KL'),
    q('c.md', '简单', false, 'PPO/clip'),
    q('d.md', '中等', true, 'GRPO/采样'),
  ]

  it('高频前置,桶内按 简单→中等→困难', () => {
    expect(sortQuestions(list).map((x) => x.file)).toEqual(['d.md', 'b.md', 'c.md', 'a.md'])
  })

  it('按 topic 第一段分组,保留传入顺序与原下标', () => {
    const groups = groupByTopic(sortQuestions(list), (x) => x.meta.topic)
    expect(groups.map((g) => g.name)).toEqual(['GRPO', 'PPO'])
    expect(groups[0].items.map((x) => x.item.file)).toEqual(['d.md', 'b.md', 'a.md'])
    expect(groups[1].items[0].index).toBe(2) // c.md 在排序后数组里的下标
  })

  it('无 topic 的题归入「未分层」', () => {
    const groups = groupByTopic([q('x.md', '简单', false, '')], (x) => x.meta.topic)
    expect(groups[0].name).toBe('未分层')
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
