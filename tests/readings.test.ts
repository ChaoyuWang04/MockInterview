import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import {
  checkReadingIndex,
  countReadings,
  getReading,
  isValidReading,
  listReadingTopicDirs,
  listReadingTopics,
  listReadings,
  parseReadingIndex,
  readingsRoot,
} from '../lib/readings'
import { listReportCompanies, listReports, reportsRoot } from '../lib/reports'

const temporaryRoots: string[] = []

const article = (title: string, date: string) =>
  `# ${title}\n\n<!-- release-date: ${date} -->\n\n完整正文。\n`

const indexOf = (...blocks: string[]) => ['# 日常研读库存', '', ...blocks, ''].join('\n')

const topicBlock = (title: string, rows: string[]) =>
  [`## ${title}`, '', '| 材料 | 机构 | 一句话 |', '|---|---|---|', ...rows, ''].join('\n')

/** 方向就是目录;_src 放原件,以 _ 开头,不该被当成方向 */
function makeReadingRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'interviewprep-readings-'))
  temporaryRoots.push(root)
  fs.mkdirSync(path.join(root, '训练方法与强化学习'))
  fs.mkdirSync(path.join(root, '图像、视频与 3D 生成'))
  fs.mkdirSync(path.join(root, '_src', '训练方法与强化学习'), { recursive: true })
  fs.writeFileSync(
    path.join(root, '训练方法与强化学习', 'GRPO.md'),
    article('GRPO 解读', '2024-02-05'),
  )
  fs.writeFileSync(path.join(root, '训练方法与强化学习', '_未发布.md'), '# 草稿\n')
  fs.writeFileSync(path.join(root, '_src', '训练方法与强化学习', 'GRPO.pdf'), 'not markdown')
  fs.writeFileSync(
    path.join(root, '图像、视频与 3D 生成', 'SegmentAnything.md'),
    article('Segment Anything 解读', '2023-04-05'),
  )
  fs.writeFileSync(
    path.join(root, 'index.md'),
    indexOf(
      topicBlock('训练方法与强化学习', [
        '| GRPO | DeepSeek | 组内相对优势,省掉 critic |',
        '| PPO | OpenAI | 尚未解读 |',
      ]),
      topicBlock('图像、视频与 3D 生成', ['| SegmentAnything | Meta | 可提示分割 |']),
      topicBlock('音频', ['| 还没写 | Kyutai | 方向已建、目录还不存在 |']),
    ),
  )
  return root
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

describe('日常研读库', () => {
  it('方向目录不含 _src 与下划线草稿', () => {
    const root = makeReadingRoot()
    expect(listReadingTopicDirs(root)).toEqual(['图像、视频与 3D 生成', '训练方法与强化学习'])
  })

  it('只列已发布文章,下划线草稿不算', () => {
    const root = makeReadingRoot()
    expect(listReadings('训练方法与强化学习', root).map((r) => r.slug)).toEqual(['GRPO'])
    expect(countReadings(root)).toBe(2)
  })

  it('文章带上方向与机构,正文里剥掉 release-date', () => {
    const root = makeReadingRoot()
    const reading = getReading('训练方法与强化学习', 'GRPO', root)
    expect(reading).toMatchObject({
      slug: 'GRPO',
      title: 'GRPO 解读',
      releaseDate: '2024-02-05',
      topic: '训练方法与强化学习',
      org: 'DeepSeek',
    })
    expect(reading?.content).not.toContain('release-date')
    expect(reading?.content).toContain('完整正文')
  })

  it('索引按方向分组,未解读行照常保留', () => {
    const root = makeReadingRoot()
    const topics = parseReadingIndex(root)
    expect(topics.map((t) => t.title)).toEqual([
      '训练方法与强化学习',
      '图像、视频与 3D 生成',
      '音频',
    ])
    expect(topics[0].entries.map((e) => e.slug)).toEqual(['GRPO', 'PPO'])
  })

  it('真实的 readings/ 与索引一致', () => {
    expect(checkReadingIndex()).toEqual([])
  })

  it('页面只给出有已发布文章的方向', () => {
    const root = makeReadingRoot()
    const topics = listReadingTopics(root)
    expect(topics.map((t) => t.title)).toEqual(['训练方法与强化学习', '图像、视频与 3D 生成'])
    expect(topics[0].readings[0].org).toBe('DeepSeek')
  })

  it('目录没有对应的 `## 方向` 时报错', () => {
    const root = makeReadingRoot()
    fs.mkdirSync(path.join(root, '音频学'))
    fs.writeFileSync(path.join(root, '音频学', 'Moshi.md'), article('Moshi 解读', '2024-09-18'))
    expect(checkReadingIndex(root)).toContain(
      'readings/音频学/ 是目录,但 index.md 里没有同名的 `## 音频学`',
    )
  })

  it('文章登记在别的方向下时报错,并指出登记在哪', () => {
    const root = makeReadingRoot()
    fs.writeFileSync(path.join(root, '音频'), '')
    fs.rmSync(path.join(root, '音频'))
    fs.renameSync(
      path.join(root, '训练方法与强化学习', 'GRPO.md'),
      path.join(root, '图像、视频与 3D 生成', 'GRPO.md'),
    )
    expect(checkReadingIndex(root)).toContain(
      '图像、视频与 3D 生成/GRPO.md 在 index.md 登记在方向「训练方法与强化学习」下,与目录不一致',
    )
  })

  it('未登记的文章报错', () => {
    const root = makeReadingRoot()
    fs.writeFileSync(
      path.join(root, '训练方法与强化学习', 'DPO.md'),
      article('DPO 解读', '2023-05-29'),
    )
    expect(checkReadingIndex(root)).toContain('训练方法与强化学习/DPO.md 未在 index.md 登记')
  })

  it('已发布行不是从新到旧时报错', () => {
    const root = makeReadingRoot()
    fs.writeFileSync(
      path.join(root, '训练方法与强化学习', 'PPO.md'),
      article('PPO 解读', '2026-01-01'),
    )
    expect(checkReadingIndex(root)).toEqual([
      '训练方法与强化学习: PPO (2026-01-01) 排在 GRPO (2024-02-05) 后面,已发布卡片必须按首发日从新到旧',
    ])
  })

  it('同一材料名不能在两个方向下各登记一次', () => {
    const root = makeReadingRoot()
    fs.writeFileSync(
      path.join(root, 'index.md'),
      indexOf(
        topicBlock('训练方法与强化学习', ['| GRPO | DeepSeek | 一 |']),
        topicBlock('图像、视频与 3D 生成', ['| GRPO | Meta | 二 |']),
      ),
    )
    expect(() => parseReadingIndex(root)).toThrow(/GRPO 重复登记/)
  })

  it('不接受路径穿越', () => {
    const root = makeReadingRoot()
    expect(isValidReading('..', 'index', root)).toBe(false)
    expect(getReading('../..', 'package', root)).toBeNull()
    expect(listReadings('../reports', root)).toEqual([])
  })
})

describe('两个解读库互斥', () => {
  it('同一篇材料不能同时出现在报告解读和日常研读里', () => {
    const reports = new Set(
      listReportCompanies(reportsRoot()).flatMap((company) =>
        listReports(company, reportsRoot()).map((report) => report.slug),
      ),
    )
    const both = listReadingTopicDirs(readingsRoot())
      .flatMap((topic) => listReadings(topic, readingsRoot()).map((r) => r.slug))
      .filter((slug) => reports.has(slug))
    expect(both).toEqual([])
  })
})
