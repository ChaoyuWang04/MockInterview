import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'

/**
 * 06-开源解读流程.md 里能机械判定的那几条,固定跑。
 *
 * 只收「判断标准唯一、误报接近零」的检查。刻意不收的四项及理由:
 *   中文数字量词  —— 「三个进程」「三层缓存」是中文习惯,实测 32/35 页命中,全是误报
 *   alt 是否一句完整的话 —— 阈值主观
 *   调参表单元格非空 —— 「—」是合法占位
 *   SVG 文字越界与重叠 —— 要浏览器 DOM,属于画图时的一次性验收,不拖慢 npm test
 */

const ROOT = path.join(__dirname, '..', 'opensource')

/** 讲稿转录的历史例外,永久豁免(06 第 1 组) */
const EXEMPT = new Set(['vllm'])

/**
 * 返工清单:2026-09-20 立新标准之前写的章页,原理段里还带着参数名。
 * **返工一章就从这里删一行**,清单空了就把这个常量和它的引用一起删掉。
 * 只豁免「原理段无标识符」这一条,其余检查照常约束它们。
 */
const PRINCIPLE_TODO = new Set([
  'sglang/02-调度器:缓存感知与零开销重叠.md',
  'sglang/04-RadixAttention:前缀缓存.md',
  'sglang/06-注意力后端.md',
  'sglang/07-执行与 CUDA graph.md',
  'sglang/08-推测解码.md',
  'sglang/09-并行:TP、PP、CP 与 DCP.md',
  'sglang/10-大规模 MoE:EP、DeepEP、DP attention、TBO 与 EPLB.md',
  'sglang/11-PD 分离与 EPD.md',
  'sglang/12-量化.md',
  'sglang/14-多模态.md',
  'sglang/15-LoRA.md',
  'sglang/16-模型加载与权重缓存.md',
  'sglang/17-RL 训推一体:权重同步与显存让出.md',
  'sglang/18-Model Gateway:跨实例的缓存感知路由.md',
])

type Page = { project: string; file: string; text: string }

function chapterPages(): Page[] {
  if (!fs.existsSync(ROOT)) return []
  const out: Page[] = []
  for (const topic of fs.readdirSync(ROOT)) {
    const tDir = path.join(ROOT, topic)
    if (!fs.statSync(tDir).isDirectory()) continue
    for (const project of fs.readdirSync(tDir)) {
      if (EXEMPT.has(project)) continue
      const pDir = path.join(tDir, project)
      if (!fs.statSync(pDir).isDirectory()) continue
      for (const file of fs.readdirSync(pDir)) {
        // 只查章页:00 总览与 99 代码索引结构不同,下划线开头的是底稿
        if (!/^\d\d-/.test(file) || !file.endsWith('.md')) continue
        if (/^(00|99)-/.test(file)) continue
        out.push({ project, file, text: fs.readFileSync(path.join(pDir, file), 'utf8') })
      }
    }
  }
  return out
}

const PAGES = chapterPages()
const label = (p: Page) => `${p.project}/${p.file}`
/** 原理段 = 一到四段;第 5、6 段才允许出现参数名 */
const principle = (t: string) =>
  t.includes('## 一、') && t.includes('## 五、') ? t.slice(t.indexOf('## 一、'), t.indexOf('## 五、')) : ''

describe('开源解读页的硬约束', () => {
  it('有章页可查', () => {
    expect(PAGES.length).toBeGreaterThan(0)
  })

  it('六段齐全', () => {
    const bad = PAGES.filter((p) => !['一', '二', '三', '四', '五', '六'].every((c) => p.text.includes(`## ${c}、`)))
    expect(bad.map(label)).toEqual([])
  })

  it('原理段(一到四)不出现反引号标识符', () => {
    const bad = PAGES.filter((p) => !PRINCIPLE_TODO.has(label(p)) && principle(p.text).includes('`'))
    expect(bad.map(label)).toEqual([])
  })

  it('返工清单只减不增:清单里的页确实还没返工', () => {
    const done = [...PRINCIPLE_TODO].filter((k) => {
      const p = PAGES.find((x) => label(x) === k)
      return p && !principle(p.text).includes('`')
    })
    expect(done, '这些页已经合标了,把它们从 PRINCIPLE_TODO 删掉').toEqual([])
  })

  it('不放代码块', () => {
    const bad = PAGES.filter((p) => p.text.includes('```'))
    expect(bad.map(label)).toEqual([])
  })

  it('正文不写文件行号(行号进底稿,符号名进代码索引页)', () => {
    const bad = PAGES.filter((p) => /`[^`]*\.(py|rs|mjs|ts|tsx|mdx):\d+/.test(p.text))
    expect(bad.map(label)).toEqual([])
  })

  it('不写模糊数量', () => {
    const bad = PAGES.filter((p) => /几[十百千万亿]|数[十百千万](?!据)/.test(p.text))
    expect(bad.map(label)).toEqual([])
  })

  it('不写施工状态', () => {
    const words = ['待确认', '探路稿', '把握程度', '候选章', '此处应有图', '下一批补']
    const bad = PAGES.flatMap((p) => words.filter((w) => p.text.includes(w)).map((w) => `${label(p)}: ${w}`))
    expect(bad).toEqual([])
  })

  it('每张表各行列数一致', () => {
    const bad = PAGES.flatMap((p) =>
      (p.text.match(/(?:^\|.*\n)+/gm) ?? [])
        .filter((block) => new Set(block.trim().split('\n').map((r) => r.split('|').length)).size > 1)
        .map(() => label(p)),
    )
    expect([...new Set(bad)]).toEqual([])
  })
})
