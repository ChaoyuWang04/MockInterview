import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  getAllNotes, getNote, isValidSlug, listAllProblems, listExtra, listHighFreq, listHot100,
  saveNote, setHighFreq,
} from '../lib/leetcode'
import { sortProblems } from '../lib/sorting'

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

  it('白名单校验(含补充题)', () => {
    expect(isValidSlug('two-sum')).toBe(true)
    expect(isValidSlug('sort-an-array')).toBe(true) // extra.md 里的补充题
    expect(isValidSlug('../../etc/passwd')).toBe(false)
    expect(isValidSlug('not-in-any-list')).toBe(false)
  })
})

describe('补充题 extra.md 与合并', () => {
  it('补充题都带 extra 标记、字段合规', () => {
    const bad = listExtra()
      .flatMap((g) => g.problems)
      .flatMap((p) => {
        const problems: string[] = []
        if (!p.extra) problems.push('缺 extra 标记')
        if (!/^\d+$/.test(p.id)) problems.push('题号非数字')
        if (!/^[a-z0-9-]+$/.test(p.slug)) problems.push(`slug 不合规: ${p.slug}`)
        if (!['简单', '中等', '困难'].includes(p.difficulty)) problems.push(`难度不合规: ${p.difficulty}`)
        return problems.map((x) => `${p.id} ${p.title}: ${x}`)
      })
    expect(bad).toEqual([])
  })

  it('同名分组并入 hot100,新分组追加在末尾,总数 = 100 + 补充数', () => {
    const hot = listHot100()
    const extra = listExtra()
    const all = listAllProblems()
    const extraTotal = extra.reduce((s, g) => s + g.problems.length, 0)

    expect(all.reduce((s, g) => s + g.problems.length, 0)).toBe(100 + extraTotal)
    // 链表组在 hot100 里已存在 → 应并入而非新增分组
    expect(all.filter((g) => g.name === '链表').length).toBe(1)
    expect(all.find((g) => g.name === '链表')!.problems.some((p) => p.slug === 'reorder-list')).toBe(true)
    // hot100 里没有「排序与字符串」→ 作为新分组追加在末尾
    expect(all.length).toBe(hot.length + 1)
    expect(all[all.length - 1].name).toBe('排序与字符串')
  })

  it('全库 slug 不重复(hot100 与补充题之间也不重复)', () => {
    const slugs = listAllProblems().flatMap((g) => g.problems.map((p) => p.slug))
    expect(new Set(slugs).size).toBe(slugs.length)
  })

  it('高频标记里的 slug 都在清单内', () => {
    const known = new Set(listAllProblems().flatMap((g) => g.problems.map((p) => p.slug)))
    expect(listHighFreq().filter((s) => !known.has(s))).toEqual([])
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

  it('高频标记增删可读回,按清单顺序写盘', () => {
    expect(listHighFreq(root)).toEqual([])
    setHighFreq('3sum', true, root) // 双指针组,清单里排在 two-sum 之后
    setHighFreq('two-sum', true, root)
    expect(listHighFreq(root)).toEqual(['two-sum', '3sum'])
    setHighFreq('two-sum', false, root)
    expect(listHighFreq(root)).toEqual(['3sum'])
  })
})

describe('排序规则', () => {
  const p = (slug: string, difficulty: string) => ({ id: '1', title: slug, slug, difficulty })
  const list = [p('a', '中等'), p('b', '困难'), p('c', '简单'), p('d', '中等')]

  it('无高频时按 简单→中等→困难,同难度保持原序', () => {
    expect(sortProblems(list, new Set()).map((x) => x.slug)).toEqual(['c', 'a', 'd', 'b'])
  })

  it('全部高频时同样按难度排序', () => {
    expect(sortProblems(list, new Set(['a', 'b', 'c', 'd'])).map((x) => x.slug)).toEqual([
      'c', 'a', 'd', 'b',
    ])
  })

  it('高频整体前置,两桶内部各自按难度排序', () => {
    // 高频:b(困难) d(中等) → d,b;非高频:a(中等) c(简单) → c,a
    expect(sortProblems(list, new Set(['b', 'd'])).map((x) => x.slug)).toEqual(['d', 'b', 'c', 'a'])
  })
})
