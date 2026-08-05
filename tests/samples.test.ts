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

  // topic/summary 是去重索引(npm run topics)的数据源,缺失或超长都会让索引失效
  it('每题都有合规的 topic 与 summary', () => {
    const bad = listCategories()
      .flatMap((c) => loadCategory(c))
      .filter((q) => !q.error)
      .flatMap((q) => {
        const problems: string[] = []
        if (!q.meta.topic) problems.push('缺 topic')
        else {
          const segs = q.meta.topic.split('/')
          if (segs.length > 2 || segs.some((s) => !s.trim())) problems.push(`topic 格式不合规: ${q.meta.topic}`)
        }
        if (!q.meta.summary) problems.push('缺 summary')
        else if (q.meta.summary.length > 50) problems.push(`summary 超 50 字(${q.meta.summary.length})`)
        return problems.map((p) => `${q.category}/${q.file}: ${p}`)
      })
    expect(bad).toEqual([])
  })
})
