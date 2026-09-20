import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
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

  it('每个项目都有总览且排在首位', () => {
    const bad = listOsTopics()
      .flatMap((t) => listOsProjects(t).map((p) => ({ t, p })))
      .filter(({ t, p }) => getOsPages(t, p)[0]?.title !== '总览')
      .map(({ t, p }) => `${t}/${p}`)
    expect(bad).toEqual([])
  })

  it('白名单校验拒绝路径穿越', () => {
    expect(isValidOsProject('..', 'sglang')).toBe(false)
    expect(isValidOsProject('推理服务', '../sglang')).toBe(false)
    expect(getOsPages('推理服务', '不存在的项目')).toEqual([])
  })
})

// 解读页引用的图片与交互版文件必须真实存在于 public/;每份规格都要有导出的产物
describe('开源解读的图片产物', () => {
  const root = path.join(process.cwd(), 'opensource')
  it('页面引用的 /opensource/<项目>/... 文件都存在,规格都有 svg 与 html', () => {
    const bad: string[] = []
    for (const t of listOsTopics()) {
      for (const p of listOsProjects(t)) {
        const dir = path.join(root, t, p)
        const pub = path.join(process.cwd(), 'public', 'opensource', p)
        for (const pg of getOsPages(t, p)) {
          for (const m of pg.content.matchAll(/\]\((\/opensource\/([^/)]+)\/([^)]+))\)/g)) {
            if (m[2] !== p) bad.push(`${p}/${pg.file}: 引用了别的项目 ${m[1]}`)
            else if (!fs.existsSync(path.join(pub, m[3]))) bad.push(`${p}/${pg.file}: ${m[1]} 不存在`)
          }
        }
        const dDir = path.join(dir, 'diagrams')
        if (!fs.existsSync(dDir)) continue
        for (const f of fs.readdirSync(dDir).filter((x) => x.endsWith('.json') && !x.startsWith('_'))) {
          const slug = f.replace(/\.json$/, '')
          for (const ext of ['.svg', '.html']) {
            if (!fs.existsSync(path.join(pub, slug + ext))) bad.push(`${p}/diagrams/${f}: 缺 ${slug}${ext}`)
          }
        }
      }
    }
    expect(bad).toEqual([])
  })
})
