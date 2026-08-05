import { describe, it, expect } from 'vitest'
import { getStats, listCategories, loadCategory } from '../lib/questions'

// 对仓库真实题库的回归防护:任何解析失败的题都会让这里报出文件名
describe('真实题库 questions/', () => {
  it('所有题目可解析,无 error', () => {
    const broken = listCategories()
      .flatMap((c) => loadCategory(c))
      .filter((q) => q.error)
      .map((q) => `${q.category}/${q.file}: ${q.error}`)
    expect(broken).toEqual([])
  })

  it('统计非空', () => {
    const s = getStats()
    expect(s.total).toBeGreaterThan(0)
    expect(s.categories.length).toBeGreaterThan(0)
  })
})
