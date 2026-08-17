import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { getAllNotes, getNote, isValidSlug, listHot100, saveNote } from '../lib/leetcode'

describe('真实清单 leetcode/hot100.md', () => {
  const groups = listHot100()

  it('恰好 100 题、17 个分组', () => {
    expect(groups.length).toBe(17)
    expect(groups.reduce((s, g) => s + g.problems.length, 0)).toBe(100)
  })

  it('每题字段合规:题号为数字、slug 为小写短横、难度三选一', () => {
    const bad = groups.flatMap((g) =>
      g.problems.flatMap((p) => {
        const problems: string[] = []
        if (!/^\d+$/.test(p.id)) problems.push('题号非数字')
        if (!p.title) problems.push('缺标题')
        if (!/^[a-z0-9-]+$/.test(p.slug)) problems.push(`slug 不合规: ${p.slug}`)
        if (!['简单', '中等', '困难'].includes(p.difficulty)) problems.push(`难度不合规: ${p.difficulty}`)
        return problems.map((x) => `${p.id} ${p.title}: ${x}`)
      }),
    )
    expect(bad).toEqual([])
  })

  it('slug 与题号无重复', () => {
    const slugs = groups.flatMap((g) => g.problems.map((p) => p.slug))
    const ids = groups.flatMap((g) => g.problems.map((p) => p.id))
    expect(new Set(slugs).size).toBe(100)
    expect(new Set(ids).size).toBe(100)
  })

  it('白名单校验', () => {
    expect(isValidSlug('two-sum')).toBe(true)
    expect(isValidSlug('../../etc/passwd')).toBe(false)
    expect(isValidSlug('not-in-hot100')).toBe(false)
  })
})

describe('笔记读写', () => {
  let root: string

  beforeEach(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'lc-'))
    fs.copyFileSync(
      path.join(process.cwd(), 'leetcode', 'hot100.md'),
      path.join(root, 'hot100.md'),
    )
  })
  afterEach(() => fs.rmSync(root, { recursive: true, force: true }))

  it('写入后可读回,内容一致', () => {
    saveNote('two-sum', '哈希表存 target-x', root)
    expect(getNote('two-sum', root).trim()).toBe('哈希表存 target-x')
    expect(getAllNotes(root)['two-sum'].trim()).toBe('哈希表存 target-x')
  })

  it('保存空内容删除笔记文件', () => {
    saveNote('two-sum', '临时', root)
    saveNote('two-sum', '', root)
    expect(getNote('two-sum', root)).toBe('')
    expect(getAllNotes(root)).toEqual({})
  })

  it('无笔记时读取返回空串,不抛异常', () => {
    expect(getNote('3sum', root)).toBe('')
    expect(getAllNotes(root)).toEqual({})
  })
})
