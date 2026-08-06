import { describe, it, expect } from 'vitest'
import { getArticle, listArticleTopics, listKbCategories } from '../lib/knowledge'

// 对仓库真实知识库的回归防护
describe('真实知识库 knowledge/', () => {
  it('所有文章可读且非空', () => {
    const empty = listKbCategories()
      .flatMap((c) => listArticleTopics(c).map((t) => ({ c, t })))
      .filter(({ c, t }) => !(getArticle(c, t) ?? '').trim())
      .map(({ c, t }) => `${c}/${t}`)
    expect(empty).toEqual([])
  })

  it('GRPO 样板文章存在', () => {
    expect(getArticle('RL', 'GRPO')).toContain('Group Relative Policy Optimization')
  })

  it('白名单校验拒绝路径穿越', () => {
    expect(getArticle('..', 'GRPO')).toBeNull()
    expect(getArticle('RL', '../GRPO')).toBeNull()
    expect(getArticle('RL', '不存在的主题')).toBeNull()
  })
})
