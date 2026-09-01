import { describe, it, expect } from 'vitest'
import {
  articleHref, findArticle, flattenArticles, getArticleBySegments, kbLinksFor, listKbTree, stripOrder,
} from '../lib/knowledge'
import { listCategories, loadCategory } from '../lib/questions'

const tree = listKbTree()
const articles = flattenArticles(tree)

describe('真实知识库 knowledge/', () => {
  it('所有文章可读且非空', () => {
    const empty = articles.filter((a) => !(getArticleBySegments(a.segments) ?? '').trim())
    expect(empty.map((a) => a.segments.join('/'))).toEqual([])
  })

  it('文章名全局唯一(题目 topic 靠文件名全库匹配,重名会指向错的文章)', () => {
    const seen = new Map<string, string[]>()
    for (const a of articles) seen.set(a.title, [...(seen.get(a.title) ?? []), a.segments.join('/')])
    const dup = [...seen.entries()].filter(([, paths]) => paths.length > 1)
    // 各章的 00-总览 允许重名:它们是 hub 页,不参与 topic 匹配
    expect(dup.filter(([title]) => title !== '总览')).toEqual([])
  })

  it('六个顶层章节按训练流程排序', () => {
    expect(tree.folders.map((f) => f.title)).toEqual([
      '模型结构', '预训练与微调', '强化学习', 'Infra', '多模态', '应用',
    ])
  })

  it('嵌套子领域可被读到(三层路径)', () => {
    expect(getArticleBySegments(['05-多模态', '01-视觉理解', 'VLM结构.md'])).toContain('#')
    expect(getArticleBySegments(['04-Infra', '01-原理', 'FlashAttention.md'])).toContain('FlashAttention')
  })

  it('占位稿被正确识别', () => {
    const ph = articles.filter((a) => a.placeholder)
    expect(ph.length).toBeGreaterThan(0)
    expect(articles.find((a) => a.title === 'GRPO')!.placeholder).toBe(false)
  })

  it('白名单校验拒绝路径穿越', () => {
    expect(getArticleBySegments(['..', 'GRPO'])).toBeNull()
    expect(getArticleBySegments(['03-强化学习', '../GRPO'])).toBeNull()
    expect(getArticleBySegments(['不存在的章', 'X'])).toBeNull()
  })

  it('stripOrder 去掉排序前缀', () => {
    expect(stripOrder('01-模型结构')).toBe('模型结构')
    expect(stripOrder('GRPO')).toBe('GRPO')
  })
})

describe('题目 topic → 文章的全库匹配', () => {
  it('findArticle 按文章名跨章节命中', () => {
    const a = findArticle('GRPO')!
    expect(a.segments).toEqual(['03-强化学习', 'GRPO.md'])
    expect(articleHref(a)).toBe('/kb/03-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/GRPO.md')
    expect(findArticle('不存在的文章')).toBeNull()
  })

  it('题库里已有文章的 topic 都能拿到链接', () => {
    const topics = listCategories().flatMap((c) =>
      loadCategory(c).map((q) => q.meta.topic?.split('/')[0]?.trim() ?? ''),
    )
    const links = kbLinksFor(topics)
    // 解码策略 已迁到 Infra 章,跨章匹配仍应成立
    expect(links['解码策略']).toBe('/kb/04-Infra/01-%E5%8E%9F%E7%90%86/%E8%A7%A3%E7%A0%81%E7%AD%96%E7%95%A5.md')
  })
})
