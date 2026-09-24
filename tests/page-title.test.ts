import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const appRoot = path.resolve(import.meta.dirname, '../app')

function walkPages(dir: string): string[] {
  const out: string[] = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...walkPages(full))
    else if (entry.name === 'page.tsx') out.push(full)
  }
  return out
}

describe('浏览器 tab 标题', () => {
  it('根布局用 template,未知路由回落刷题系统', () => {
    const layout = fs.readFileSync(path.join(appRoot, 'layout.tsx'), 'utf8')
    expect(layout).toContain("default: '刷题系统'")
    expect(layout).toContain("template: '%s'")
  })

  it('每个 page.tsx 都导出 metadata 或 generateMetadata', () => {
    const missing: string[] = []
    for (const file of walkPages(appRoot)) {
      const text = fs.readFileSync(file, 'utf8')
      if (!/export const metadata: Metadata = \{ title:/.test(text) && !text.includes('export async function generateMetadata')) {
        missing.push(path.relative(appRoot, file))
      }
    }
    expect(missing).toEqual([])
  })

  it('固定入口用批准的短名', () => {
    const read = (rel: string) => fs.readFileSync(path.join(appRoot, rel), 'utf8')
    expect(read('page.tsx')).toContain("title: '主页'")
    expect(read('interview/page.tsx')).toContain("title: '模拟面试'")
    expect(read('interview/sessions/page.tsx')).toContain("title: '模拟面试'")
    expect(read('kb/page.tsx')).toContain("title: '知识库'")
    expect(read('opensource/page.tsx')).toContain("title: '开源解读'")
    expect(read('reports/page.tsx')).toContain("title: '报告解读'")
    expect(read('readings/page.tsx')).toContain("title: '日常研读'")
    expect(read('leetcode/page.tsx')).toContain("title: '立扣'")
  })
})
