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
    const bad = PAGES.filter((p) => principle(p.text).includes('`'))
    expect(bad.map(label)).toEqual([])
  })

  /** 反引号好查,裸写的一样是名字:--chunked-prefill-size、SGLANG_XXX、VLLM_XXX、mem_fraction_static */
  it('原理段(一到四)也不出现裸参数名', () => {
    // 前一个字符不能是标识符的一部分;中文标点、句读、行首都算边界
    const BARE = /(?<![A-Za-z0-9_/.\-])(--[a-z][a-z0-9-]{4,}|(?:SGLANG|VLLM)_[A-Z0-9_]{3,}|[a-z][a-z0-9]*(?:_[a-z0-9]+){1,})/g
    const bad = PAGES.flatMap((p) => {
      const hits = [...new Set([...principle(p.text).matchAll(BARE)].map((m) => m[1]))]
      return hits.map((h) => `${label(p)}: ${h}`)
    })
    expect(bad).toEqual([])
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

  /**
   * 跨章重复:并行派出去的子 agent 互相看不见,同一个机制很容易被两章各讲一遍。
   * 只拦「一模一样的长句」——概念层面的重复要人判断,在主线程验收时看。
   * 页头回指、典型配置的免责句这类结构性重复是设计如此,放行。
   */
  it('同一项目内不同章没有一模一样的长句', () => {
    // 结构性重复,设计如此:页头回指、典型配置的免责句、调参入口那句
    const BOILER = [/^每一条对应源码的哪个文件与符号/, /^这一页只说/, /不是实测最优/, /能直接抄走的起法/, /^这一页的参数全是启动参数/, /^说法都在源码基准/, /^这一页的参数全是启动配置/, /^对照对象是/]
    const seen = new Map<string, Set<string>>()
    for (const p of PAGES) {
      const body = p.text.replace(/!\[.*?\]\(.*?\)/g, '').replace(/\|.*?\|/g, '')
      for (const raw of body.split(/[。;\n]/)) {
        const s = raw.replace(/[\s*`—·]/g, '').trim()
        if (s.length < 20 || BOILER.some((b) => b.test(s))) continue
        if (!seen.has(s)) seen.set(s, new Set())
        seen.get(s)!.add(label(p))
      }
    }
    const bad = [...seen.entries()].filter(([, v]) => v.size > 1).map(([s, v]) => `${[...v].join(' / ')}: ${s.slice(0, 40)}`)
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
