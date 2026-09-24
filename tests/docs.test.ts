import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const projectRoot = path.resolve(import.meta.dirname, '..')
const docsRoot = path.join(projectRoot, 'docs')

function activeMarkdownFiles(): string[] {
  const files = [
    path.join(projectRoot, 'README.md'),
    path.join(projectRoot, 'AGENTS.md'),
    path.join(projectRoot, 'CLAUDE.md'),
  ]

  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'superpowers') continue
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) walk(full)
      else if (entry.name.endsWith('.md')) files.push(full)
    }
  }

  walk(docsRoot)
  return files
}

describe('活动文档', () => {
  it('本地 Markdown 链接都指向存在的文件', () => {
    const broken: string[] = []

    for (const file of activeMarkdownFiles()) {
      const text = fs.readFileSync(file, 'utf8')
      for (const match of text.matchAll(/!?(?:\[[^\]]*\])\(([^)]+)\)/g)) {
        const raw = match[1].trim().replace(/^<|>$/g, '')
        if (/^(?:https?:|mailto:|#)/.test(raw) || raw.startsWith('/')) continue
        const target = decodeURIComponent(raw.split('#')[0])
        if (!target || fs.existsSync(path.resolve(path.dirname(file), target))) continue
        broken.push(`${path.relative(projectRoot, file)} -> ${raw}`)
      }
    }

    expect(broken).toEqual([])
  })

  // AGENTS.md 是唯一真源;CLAUDE.md 只导入它,两份手工副本迟早漂移
  it('CLAUDE.md 只导入 AGENTS.md,入口指向全景地图', () => {
    const agents = fs.readFileSync(path.join(projectRoot, 'AGENTS.md'), 'utf8')
    const claude = fs.readFileSync(path.join(projectRoot, 'CLAUDE.md'), 'utf8')

    expect(claude.trim()).toBe('@AGENTS.md')
    expect(agents).toContain('docs/00-START.md')
  })

  it('Codex 与 Claude 读同一份 skill', () => {
    const link = path.join(projectRoot, '.agents/skills')
    expect(fs.lstatSync(link).isSymbolicLink()).toBe(true)
    expect(fs.readlinkSync(link)).toBe('../.claude/skills')
  })

  it('入口统一为七个功能模块,运行设施不算模块', () => {
    const entryFiles = ['README.md', 'AGENTS.md', 'CLAUDE.md', 'docs/00-START.md']
    const entry = entryFiles
      .map((file) => fs.readFileSync(path.join(projectRoot, file), 'utf8'))
      .join('\n')

    expect(entry).toContain('七个功能模块')
    expect(entry).not.toContain('五个功能模块')
    expect(entry).not.toContain('六个功能模块')
  })

  it('入口地图覆盖全部现行手册', () => {
    const start = fs.readFileSync(path.join(docsRoot, '00-START.md'), 'utf8')
    for (const manual of [
      '01-TASK.md',
      '02-题库导入流程.md',
      '03-题目写作规范.md',
      '04-知识库地图.md',
      '05-知识库写作契约.md',
      '06-开源解读流程.md',
      '07-LeetCode清单.md',
      '08-常驻服务.md',
      '09-日常维护.md',
      '10-材料解读流程.md',
      '11-模拟面试系统.md',
      '12-范式库维护.md',
      '13-周会纪要.md',
      '14-开源贡献.md',
    ]) {
      expect(start, `00-START 缺少 ${manual}`).toContain(manual)
    }
  })

  it('主页把报告解读作为开源项目旁边的独立入口', () => {
    const home = fs.readFileSync(path.join(projectRoot, 'app/page.tsx'), 'utf8')
    const opensourceAt = home.indexOf('href="/opensource"')
    const reportsAt = home.indexOf('href="/reports"')

    expect(opensourceAt).toBeGreaterThanOrEqual(0)
    expect(reportsAt).toBeGreaterThan(opensourceAt)
    expect(home.slice(opensourceAt, reportsAt)).not.toContain('href="/leetcode"')
    expect(home).toContain('报告解读')
  })

  it('主页把日常研读作为报告解读旁边的独立入口', () => {
    const home = fs.readFileSync(path.join(projectRoot, 'app/page.tsx'), 'utf8')
    const reportsAt = home.indexOf('href="/reports"')
    const readingsAt = home.indexOf('href="/readings"')

    expect(readingsAt).toBeGreaterThan(reportsAt)
    expect(home.slice(reportsAt, readingsAt)).not.toContain('href="/leetcode"')
    expect(home).toContain('日常研读')
  })

  it('报告解读动态路由不对 Next 参数重复解码', () => {
    const route = fs.readFileSync(
      path.join(projectRoot, 'app/reports/[company]/[report]/page.tsx'),
      'utf8',
    )

    expect(route).not.toContain('decodeURIComponent')
  })

  // 方向名全是中文,动态段是百分号编码送进来的,不解码会 404(实测)。/opensource 同理
  it('日常研读动态路由必须解码中文方向段', () => {
    const route = fs.readFileSync(
      path.join(projectRoot, 'app/readings/[topic]/[paper]/page.tsx'),
      'utf8',
    )

    expect(route).toContain('decodeURIComponent')
  })

  it('不再恢复已废弃的静态面试池说明', () => {
    const task = fs.readFileSync(path.join(docsRoot, '01-TASK.md'), 'utf8')
    const interview = fs.readFileSync(path.join(docsRoot, '11-模拟面试系统.md'), 'utf8')
    const active = `${task}\n${interview}`

    expect(active).not.toMatch(/94 道|365 条|93 项/)
  })
})
