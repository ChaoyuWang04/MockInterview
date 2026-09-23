import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { parseScanRecord, scanRecordPath, votesByArticle } from '../lib/alphaxiv'
import { listReadMarks, readMarksPath, setReadMark } from '../lib/read-marks'
import { daysBetween, groupByAge, todayInShanghai } from '../lib/timeline'
import {
  ORG_FILE,
  RECORD_FILE,
  appendRows,
  matchOrg,
  parseOrgList,
  parseScanRecord as scriptParseScanRecord,
  selectCandidates,
} from '../scripts/papers-feed.mjs'

const temporaryRoots: string[] = []
afterEach(() => {
  for (const root of temporaryRoots.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})
function tempRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'interviewprep-alphaxiv-'))
  temporaryRoots.push(root)
  return root
}

const orgs = parseOrgList(fs.readFileSync(ORG_FILE, 'utf8'))

describe('机构名单', () => {
  it('两组都有条目,目录名唯一且可作目录', () => {
    const groups = new Set(orgs.map((o: { group: string }) => o.group))
    expect(groups).toEqual(new Set(['企业与知名新兴实验室', '知名高校与研究组']))
    const dirs = orgs.map((o: { dir: string }) => o.dir)
    expect(new Set(dirs).size).toBe(dirs.length)
    for (const dir of dirs) expect(dir).toMatch(/^[A-Za-z0-9.-]+$/)
  })

  it('别名按「相等或以别名加空格、逗号开头」匹配,目录名本身不参与', () => {
    expect(matchOrg('Google DeepMind', orgs)?.dir).toBe('Google')
    expect(matchOrg('Meta Platforms, Inc.', orgs)?.dir).toBe('Meta')
    expect(matchOrg('stanford university', orgs)?.dir).toBe('Stanford')
    expect(matchOrg('Washington University in St. Louis', orgs)).toBeNull()
    expect(matchOrg('Googleplex Labs', orgs)).toBeNull()
    expect(matchOrg('Phi AI Labs', orgs)).toBeNull()
  })
})

const paper = (over: Record<string, unknown>) => ({
  universal_paper_id: '2609.00001',
  title: 'Some Paper: A Subtitle',
  first_publication_date: '2026-09-20T00:00:00.000Z',
  topics: ['Computer Science', 'agents'],
  metrics: { public_total_votes: 40 },
  organization_info: [{ name: 'NVIDIA' }],
  ...over,
})

describe('papers:feed 挑选', () => {
  const base = {
    today: '2026-09-23',
    threshold: 30,
    days: 7,
    recordIds: new Set<string>(),
    orgs,
    known: new Set<string>(['deepseekv41flash']),
  }

  it('票数线、7 天窗口、计算机方向、扫描记录去重', () => {
    const rows = selectCandidates(
      [
        paper({ universal_paper_id: 'ok' }),
        paper({ universal_paper_id: 'low', metrics: { public_total_votes: 29 } }),
        paper({ universal_paper_id: 'old', first_publication_date: '2026-09-15' }),
        paper({ universal_paper_id: 'edge', first_publication_date: '2026-09-16' }),
        paper({ universal_paper_id: 'astro', topics: ['Physics', 'astro-ph.CO'] }),
        paper({ universal_paper_id: 'cs-sub', topics: ['cs.LG'] }),
        paper({ universal_paper_id: 'seen' }),
        paper({ universal_paper_id: 'ok' }),
      ],
      { ...base, recordIds: new Set(['seen']) },
    )
    expect(rows.map((r: { id: string }) => r.id).sort()).toEqual(['cs-sub', 'edge', 'ok'])
  })

  it('初判:已在库 > 名单内 > 名单外,按票数从高到低', () => {
    const rows = selectCandidates(
      [
        paper({ universal_paper_id: 'a', metrics: { public_total_votes: 31 } }),
        paper({
          universal_paper_id: 'b',
          metrics: { public_total_votes: 90 },
          organization_info: [{ name: 'Phi AI Labs' }],
        }),
        paper({ universal_paper_id: 'c', title: 'DeepSeek-V4.1-Flash: Pushing the Limits' }),
        paper({ universal_paper_id: 'd', title: 'Pipe | in title', organization_info: [] }),
      ],
      base,
    )
    expect(rows.map((r: { id: string; verdict: string }) => [r.id, r.verdict])).toEqual([
      ['b', 'B · 名单外'],
      ['c', 'D · 已在库'],
      ['d', 'B · 名单外'],
      ['a', 'A · 名单内(NVIDIA)'],
    ])
    expect(rows[2].title).toBe('Pipe / in title')
    expect(rows[2].orgs).toBe('(alphaXiv 未标)')
  })

  it('追加到表格末尾,解析回来一致', () => {
    const text = fs.readFileSync(RECORD_FILE, 'utf8')
    const rows = selectCandidates([paper({})], base)
    const next = appendRows(text, rows)
    const parsed = parseScanRecord(next)
    expect(parsed.at(-1)).toMatchObject({ id: '2609.00001', votes: 40, destination: '待定' })
    expect(next.startsWith(text.replace(/\n+$/, ''))).toBe(true)
  })
})

describe('扫描记录与票数', () => {
  it('防漂移:页面与脚本读出同一份扫描记录', () => {
    const text = fs.readFileSync(scanRecordPath(), 'utf8')
    expect(parseScanRecord(text)).toEqual(scriptParseScanRecord(text))
  })

  it('只按去向路径把票数挂到对应库的文章上', () => {
    const repo = tempRoot()
    fs.mkdirSync(path.join(repo, 'readings'))
    fs.writeFileSync(
      scanRecordPath(repo),
      [
        '| 达标日 | 编号 | 标题 | 机构 | 票数 | 判定 | 去向 |',
        '|---|---|---|---|---|---|---|',
        '| 2026-09-22 | 1 | A | NVIDIA | 127 | A · 名单内 | reports/NVIDIA/SoL-Pi |',
        '| 2026-09-22 | 2 | B | X | 47 | B · 名单外 | `readings/自进化系统/ScienceIDE` |',
        '| 2026-09-22 | 3 | C | Y | 99 | 不收 | 不收:非计算机研究 |',
      ].join('\n'),
    )
    expect([...votesByArticle('reports', repo)]).toEqual([['NVIDIA/SoL-Pi', 127]])
    expect([...votesByArticle('readings', repo)]).toEqual([['自进化系统/ScienceIDE', 47]])
  })
})

describe('已读标记', () => {
  it('勾选只增删那一行,说明与其他行原样保留', () => {
    const root = tempRoot()
    setReadMark(root, 'NVIDIA/SoL-Pi', true)
    setReadMark(root, 'DeepSeek/DeepSeek-V2', true)
    fs.appendFileSync(readMarksPath(root), '手写备注\n')
    setReadMark(root, 'NVIDIA/SoL-Pi', true)
    expect([...listReadMarks(root)]).toEqual(['NVIDIA/SoL-Pi', 'DeepSeek/DeepSeek-V2'])
    setReadMark(root, 'NVIDIA/SoL-Pi', false)
    const text = fs.readFileSync(readMarksPath(root), 'utf8')
    expect(text).toContain('# 已读标记')
    expect(text).toContain('手写备注')
    expect([...listReadMarks(root)]).toEqual(['DeepSeek/DeepSeek-V2'])
  })

  it('文件不存在时视为全部未读', () => {
    expect(listReadMarks(tempRoot()).size).toBe(0)
  })
})

describe('按时间视图', () => {
  it('五段按首发日距今划分,边界含在较近的一段', () => {
    const today = '2026-09-23'
    const item = (key: string, releaseDate: string, votes: number | null = null) => ({
      key,
      releaseDate,
      votes,
    })
    const groups = groupByAge(
      [
        item('d3', '2026-09-20'),
        item('d4', '2026-09-19'),
        item('d7', '2026-09-16'),
        item('d30', '2026-08-24'),
        item('d31', '2026-08-23'),
        item('d182', '2026-03-25'),
        item('d183', '2026-03-24'),
        item('future', '2026-09-24'),
      ],
      today,
    )
    expect(groups.map((g) => [g.label, g.items.map((i) => i.key)])).toEqual([
      ['三天内', ['future', 'd3']],
      ['七天内', ['d4', 'd7']],
      ['一个月内', ['d30']],
      ['六个月内', ['d31', 'd182']],
      ['六个月以外', ['d183']],
    ])
  })

  it('段内先按票数从高到低,无票数的排后,再按首发日从新到旧、键名升序', () => {
    const [group] = groupByAge(
      [
        { key: 'b', releaseDate: '2026-09-22', votes: null },
        { key: 'a', releaseDate: '2026-09-22', votes: null },
        { key: 'new', releaseDate: '2026-09-23', votes: null },
        { key: 'hot', releaseDate: '2026-09-20', votes: 169 },
        { key: 'warm', releaseDate: '2026-09-22', votes: 42 },
      ],
      '2026-09-23',
    )
    expect(group.items.map((i) => i.key)).toEqual(['hot', 'warm', 'new', 'a', 'b'])
  })

  it('天数与北京时间的今天', () => {
    expect(daysBetween('2026-09-20', '2026-09-23')).toBe(3)
    expect(todayInShanghai(new Date('2026-09-22T16:30:00Z'))).toBe('2026-09-23')
  })
})

describe('papers:pull', () => {
  it('去向映射到两个库各自的原件目录,已在库与未入库的行跳过', async () => {
    const { missingSources, sourcePathOf } = await import('../scripts/papers-pull.mjs')
    const repo = tempRoot()
    expect(sourcePathOf('reports/NVIDIA/SoL-Pi', repo)).toBe(path.join(repo, 'papers', 'NVIDIA', 'SoL-Pi.pdf'))
    expect(sourcePathOf('`readings/自进化系统/X`', repo)).toBe(
      path.join(repo, 'readings', '_src', '自进化系统', 'X.pdf'),
    )
    expect(sourcePathOf('不收:非计算机研究', repo)).toBeNull()
    fs.mkdirSync(path.join(repo, 'papers', 'Xiaomi'), { recursive: true })
    fs.writeFileSync(path.join(repo, 'papers', 'Xiaomi', 'MiMo.pdf'), '%PDF')
    const row = (id: string, verdict: string, destination: string) => ({
      date: '2026-09-23', id, title: '', orgs: '', votes: 40, verdict, destination,
    })
    const missing = missingSources(
      [
        row('1', 'A · 名单内', 'reports/NVIDIA/SoL-Pi'),
        row('2', 'A · 名单内', 'reports/Xiaomi/MiMo'),
        row('3', 'D · 已在库', 'reports/DeepSeek/DeepSeek-V4.1-Flash'),
        row('4', '不收', '不收:无原件'),
      ],
      repo,
    )
    expect(missing.map((r: { id: string }) => r.id)).toEqual(['1'])
  })
})
