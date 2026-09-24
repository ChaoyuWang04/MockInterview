import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const contribRoot = path.resolve(import.meta.dirname, '..', 'contrib')
const HEADER = '| 选题 | 上游 issue | 状态 | PR | 结论 |'
const STATES = new Set(['候选', '已认领', '进行中', '待发布', '已提交', '已合入', '已关闭'])

describe('开源贡献台账', () => {
  const ledgers = fs.readdirSync(contribRoot).filter((name) => name.endsWith('.md'))

  it('每份台账都是模板的那张表,状态取自 14 的枚举', () => {
    const problems: string[] = []
    for (const name of ledgers) {
      const lines = fs.readFileSync(path.join(contribRoot, name), 'utf8').split('\n')
      const headerAt = lines.indexOf(HEADER)
      if (headerAt === -1) {
        problems.push(`${name}: 缺少表头 ${HEADER}`)
        continue
      }
      for (const row of lines.slice(headerAt + 2)) {
        if (!row.startsWith('|')) break
        const state = row.split('|')[3].trim()
        if (name !== '_template.md' && !STATES.has(state)) problems.push(`${name}: 未知状态「${state}」`)
      }
    }
    expect(problems).toEqual([])
  })

  it('台账不写私有实验仓的地址', () => {
    const leaks = ledgers.filter((name) =>
      /-lab\b|1Project\/oss|\.lab\//.test(fs.readFileSync(path.join(contribRoot, name), 'utf8')),
    )
    expect(leaks).toEqual([])
  })
})
