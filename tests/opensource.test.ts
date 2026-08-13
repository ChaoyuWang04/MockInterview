import { describe, it, expect } from 'vitest'
import { getOsPages, isValidOsProject, listOsProjects, listOsTopics } from '../lib/opensource'

// 对仓库真实解读库的回归防护
describe('真实解读库 opensource/', () => {
  it('所有解读页非空,文件名带 NN- 序号前缀', () => {
    const bad = listOsTopics()
      .flatMap((t) => listOsProjects(t).map((p) => ({ t, p })))
      .flatMap(({ t, p }) =>
        getOsPages(t, p).flatMap((pg) => {
          const problems: string[] = []
          if (!pg.content.trim()) problems.push('内容为空')
          if (!/^\d+-/.test(pg.file)) problems.push('缺 NN- 序号前缀')
          return problems.map((x) => `${t}/${p}/${pg.file}: ${x}`)
        }),
      )
    expect(bad).toEqual([])
  })

  it('LMCache 样板解读存在且首节为总览', () => {
    const pages = getOsPages('推理服务', 'LMCache')
    expect(pages.length).toBeGreaterThanOrEqual(5)
    expect(pages[0].title).toBe('总览')
  })

  it('白名单校验拒绝路径穿越', () => {
    expect(isValidOsProject('..', 'LMCache')).toBe(false)
    expect(isValidOsProject('推理服务', '../LMCache')).toBe(false)
    expect(getOsPages('推理服务', '不存在的项目')).toEqual([])
  })
})
