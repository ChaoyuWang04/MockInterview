import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import {
  countReports,
  getReport,
  isValidReport,
  listReportCompanies,
  listReports,
} from '../lib/reports'

const temporaryRoots: string[] = []

function makeReportRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'interviewprep-reports-'))
  temporaryRoots.push(root)
  fs.mkdirSync(path.join(root, 'DeepSeek'))
  fs.mkdirSync(path.join(root, 'OpenAI'))
  fs.mkdirSync(path.join(root, '_草稿'))
  fs.writeFileSync(
    path.join(root, 'DeepSeek', 'DeepSeek-V4.md'),
    '# DeepSeek-V4 Technical Report 解读\n\n完整正文。\n',
  )
  fs.writeFileSync(path.join(root, 'DeepSeek', '_未发布.md'), '# 草稿\n')
  fs.writeFileSync(path.join(root, 'OpenAI', 'GPT.md'), '# GPT Technical Report 解读\n\n完整正文。\n')
  return root
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

describe('基模报告 reports/', () => {
  it('按公司扫描已发布 Markdown,忽略隐藏目录和草稿', () => {
    const root = makeReportRoot()

    expect(listReportCompanies(root)).toEqual(['DeepSeek', 'OpenAI'])
    expect(listReports('DeepSeek', root)).toEqual([
      {
        slug: 'DeepSeek-V4',
        title: 'DeepSeek-V4 Technical Report 解读',
      },
    ])
    expect(listReports('OpenAI', root)).toEqual([
      { slug: 'GPT', title: 'GPT Technical Report 解读' },
    ])
    expect(countReports(root)).toBe(2)
  })

  it('读取一篇完整长文并拒绝路径穿越', () => {
    const root = makeReportRoot()

    expect(getReport('DeepSeek', 'DeepSeek-V4', root)?.content).toContain('完整正文')
    expect(isValidReport('..', 'DeepSeek-V4', root)).toBe(false)
    expect(isValidReport('DeepSeek', '../DeepSeek-V4', root)).toBe(false)
    expect(getReport('DeepSeek', '不存在', root)).toBeNull()
  })

  it('拒绝把空文件或缺少一级标题的文件发布到网页', () => {
    const root = makeReportRoot()
    fs.writeFileSync(path.join(root, 'DeepSeek', '缺标题.md'), '只有正文。\n')

    expect(() => listReports('DeepSeek', root)).toThrow(/缺标题\.md.*一级标题/)

    fs.unlinkSync(path.join(root, 'DeepSeek', '缺标题.md'))
    fs.writeFileSync(path.join(root, 'DeepSeek', '空正文.md'), '# 只有标题\n')
    expect(() => listReports('DeepSeek', root)).toThrow(/空正文\.md.*正文为空/)
  })

  it('公司名和报告名可以包含字面百分号', () => {
    const root = makeReportRoot()
    fs.mkdirSync(path.join(root, '100%Lab'))
    fs.writeFileSync(path.join(root, '100%Lab', 'Model%2.md'), '# Model%2\n\n完整正文。\n')

    expect(getReport('100%Lab', 'Model%2', root)?.title).toBe('Model%2')
  })

  it('仓库中的每篇已发布报告都满足标题和正文契约', () => {
    for (const company of listReportCompanies()) {
      for (const summary of listReports(company)) {
        const report = getReport(company, summary.slug)
        expect(report, `${company}/${summary.slug} 无法读取`).not.toBeNull()
        expect(report?.content).toMatch(/^#\s+.+$/m)
        expect(report?.content.replace(/^#\s+.+$/m, '').trim()).not.toBe('')
      }
    }
  })

  it('基模报告使用渲染器支持的行内数学边界', () => {
    for (const company of listReportCompanies()) {
      for (const summary of listReports(company)) {
        const content = getReport(company, summary.slug)?.content
        expect(content, `${company}/${summary.slug} 仍使用 \\(...\\)`).not.toMatch(/\\\(|\\\)/)
      }
    }
  })
})
