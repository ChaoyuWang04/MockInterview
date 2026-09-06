import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import { describe, expect, it } from 'vitest'

const projectRoot = path.resolve(import.meta.dirname, '..')

/** 五个内容目录都会被页面按 Markdown 渲染,星号泄漏对它们一视同仁 */
const contentRoots = ['reports', 'knowledge', 'opensource', 'questions', 'leetcode']

/** 与各 lib 的 isVisible 口径一致:`.` 或 `_` 开头的文件与目录不参与渲染 */
function isVisible(name: string): boolean {
  return !name.startsWith('.') && !name.startsWith('_')
}

function markdownFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) return []
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    if (!isVisible(entry.name)) return []
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return markdownFiles(full)
    return entry.isFile() && entry.name.endsWith('.md') ? [full] : []
  })
}

interface HastNode {
  type: string
  tagName?: string
  value?: string
  children?: HastNode[]
}

/** 代码块与行内代码里的 `**` 是合法内容,整棵跳过 */
const literalTags = new Set(['code', 'pre'])

/**
 * 取渲染后真正落到页面上的文本。
 * 按节点边界用换行拼接,避免相邻节点各带一个 `*` 时凑出假的 `**`。
 */
function renderedText(node: HastNode): string {
  if (node.type === 'element' && literalTags.has(node.tagName ?? '')) return ''
  if (node.type === 'text') return node.value ?? ''
  return (node.children ?? [])
    .map(renderedText)
    .filter((text) => text !== '')
    .join('\n')
}

/** 复用页面同一条 remark 链;不挂 rehype-raw,注释与裸 HTML 本来就不进正文 */
const processor = unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype)

function renderMarkdown(source: string): string {
  return renderedText(processor.runSync(processor.parse(source)) as unknown as HastNode)
}

function excerpt(text: string, at: number): string {
  const head = at > 30 ? '…' : ''
  const tail = at + 32 < text.length ? '…' : ''
  return `${head}${text.slice(Math.max(0, at - 30), at + 32).replace(/\n/g, '⏎')}${tail}`
}

describe('Markdown 星号泄漏', () => {
  it('认得出被 flanking 规则吃掉的两种加粗', () => {
    // ① 闭合 ** 夹在标点和文字之间
    expect(renderMarkdown('**ZeRO-3 与 PP 的区别:**前者把数据并行做到了参数上。')).toContain('**')
    // ② 开启 ** 夹在文字和标点之间
    expect(renderMarkdown('因为**「包一层」定义的是通信的边界**。')).toContain('**')
  })

  it('不误报代码、正常加粗和相邻节点', () => {
    expect(renderMarkdown('```python\nscale = base ** 0.5\n```')).not.toContain('**')
    expect(renderMarkdown('行内 `a ** b` 只是幂运算。')).not.toContain('**')
    expect(renderMarkdown('用 **压缩稀疏注意力(CSA)** 把细节先压缩。')).not.toContain('**')
    expect(renderMarkdown('- 结尾一个星号 \\*\n- \\* 开头一个星号')).not.toContain('**')
  })

  it('全库正文渲染后不出现字面 **', () => {
    const leaks: string[] = []

    for (const root of contentRoots) {
      for (const file of markdownFiles(path.join(projectRoot, root))) {
        // 题目的 frontmatter 不进正文,按 lib/questions.ts 的口径先摘掉
        const text = renderMarkdown(matter(fs.readFileSync(file, 'utf8')).content)
        const at = text.indexOf('**')
        if (at === -1) continue
        const count = text.split('**').length - 1
        leaks.push(`${path.relative(projectRoot, file)}(${count} 处):${excerpt(text, at)}`)
      }
    }

    // 成因与修法见 docs/09-日常维护.md「踩过的雷」的 CommonMark flanking 一条
    expect(leaks).toEqual([])
  })
})

describe('Markdown 窄屏渲染', () => {
  it('给宽表格提供局部横向滚动容器', () => {
    const renderer = fs.readFileSync(path.join(projectRoot, 'components/Markdown.tsx'), 'utf8')

    expect(renderer).toContain('markdown-table-scroll')
    expect(renderer).toMatch(/table\s*\([^)]*\)\s*\{[\s\S]*?<table\b/)
  })

  it('给宽表格和块级公式限制宽度并提供局部横向滚动', () => {
    const css = fs.readFileSync(path.join(projectRoot, 'app/globals.css'), 'utf8')

    expect(css).toMatch(/\.prose\s+\.markdown-table-scroll[\s\S]*?max-width:\s*100%/)
    expect(css).toMatch(/\.prose\s+\.katex-display[\s\S]*?max-width:\s*100%/)
    expect(css.match(/overflow-x:\s*auto/g)?.length ?? 0).toBeGreaterThanOrEqual(2)
  })
})
