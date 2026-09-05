import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const projectRoot = path.resolve(import.meta.dirname, '..')

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
