// 桌面材料分流 · 第 3 步「归档」(docs/10-材料解读流程.md 第二节)
//
// 读一份核实过的扫描报告,按「去向」列移动原件并登记索引。
// **只移动,绝不删除**——判错了能从废纸篓捞回来。
//
// 用法:
//   npm run papers:file -- <报告路径>          先干跑,只打印会做什么
//   npm run papers:file -- <报告路径> --apply  真的执行
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const REPO = process.cwd()
const PAPERS = path.join(REPO, 'papers')
const READINGS = path.join(REPO, 'readings')
const SRC = path.join(READINGS, '_src')
const TODAY = new Date().toISOString().slice(0, 10)
const TRASH = path.join(os.homedir(), '.Trash', `interviewprep-未收-${TODAY}`)

/** 解析报告里的所有表格行;只认「文件」列以反引号包着绝对路径的那些 */
function parseReport(file) {
  const rows = []
  for (const [i, line] of fs.readFileSync(file, 'utf8').split(/\r?\n/).entries()) {
    if (!line.startsWith('|')) continue
    const cells = line.split('|').slice(1, -1).map((c) => c.trim())
    if (cells.length < 5) continue
    const source = cells[0].replace(/^`|`$/g, '')
    if (!path.isAbsolute(source)) continue
    rows.push({
      at: `第 ${i + 1} 行`,
      source,
      slug: cells[1],
      target: cells[2],
      summary: cells[3],
      reason: cells[4],
    })
  }
  return rows
}

/** 把一行登记进索引对应的 `## 方向` 表格末尾。方向不存在直接报错,不擅自新建 */
function register(indexFile, topic, row) {
  const lines = fs.readFileSync(indexFile, 'utf8').split(/\r?\n/)
  const start = lines.findIndex((line) => line.trim() === `## ${topic}`)
  if (start < 0) throw new Error(`${path.relative(REPO, indexFile)} 里没有 \`## ${topic}\``)
  let last = -1
  for (let i = start + 1; i < lines.length; i++) {
    if (lines[i].startsWith('## ')) break
    if (lines[i].startsWith('|')) last = i
  }
  if (last < 0) throw new Error(`\`## ${topic}\` 下面没有表格`)
  lines.splice(last + 1, 0, row)
  fs.writeFileSync(indexFile, lines.join('\n'))
}

function appendVerdict(entry) {
  const file = path.join(READINGS, '_判定档案.md')
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
  const start = lines.findIndex((line) => line.trim() === '## 判定不收')
  if (start < 0) throw new Error('_判定档案.md 里没有 `## 判定不收`')
  let last = -1
  for (let i = start + 1; i < lines.length; i++) {
    if (lines[i].startsWith('## ')) break
    if (lines[i].startsWith('|')) last = i
  }
  lines.splice(last + 1, 0, entry)
  fs.writeFileSync(file, lines.join('\n'))
}

function plan(rows) {
  const actions = []
  const errors = []
  const seen = new Set()

  for (const row of rows) {
    const { at, source, slug, target, summary } = row
    if (target === '跳过' || target === '待定' || target === '') continue
    if (!fs.existsSync(source)) {
      errors.push(`${at}: 源文件不存在 ${source}`)
      continue
    }
    if (!slug) {
      errors.push(`${at}: 建议名为空`)
      continue
    }
    if (seen.has(slug)) {
      errors.push(`${at}: 材料名 ${slug} 在本报告里出现两次`)
      continue
    }
    seen.add(slug)

    const ext = path.extname(source)
    const notCollected = target.match(/^不收[::]\s*(.+)$/)
    const toReports = target.match(/^报告\/([^/]+)\/([^/]+)$/)
    const toReadings = target.match(/^研读\/(.+)$/)

    if (notCollected) {
      actions.push({
        kind: '不收',
        source,
        dest: path.join(TRASH, path.basename(source)),
        verdict: `| \`${slug}\` | 分流判定 | ${notCollected[1].trim()}(${TODAY}) |`,
      })
    } else if (toReports) {
      const [, topic, company] = toReports
      if (!summary) {
        errors.push(`${at}: 进库必须填「一句话」`)
        continue
      }
      actions.push({
        kind: '报告解读',
        source,
        dest: path.join(PAPERS, company, `${slug}${ext}`),
        index: path.join(REPO, 'reports', 'index.md'),
        topic,
        row: `| ${slug} | ${company} | ${summary} |`,
      })
    } else if (toReadings) {
      const topic = toReadings[1].trim()
      if (!summary) {
        errors.push(`${at}: 进库必须填「一句话」`)
        continue
      }
      const org = row.reason.match(/机构[::]\s*([^;;]+)/)?.[1]?.trim() ?? '待补'
      actions.push({
        kind: '日常研读',
        source,
        dest: path.join(SRC, topic, `${slug}${ext}`),
        index: path.join(READINGS, 'index.md'),
        topic,
        row: `| ${slug} | ${org} | ${summary} |`,
      })
    } else {
      errors.push(`${at}: 看不懂的去向「${target}」`)
      continue
    }

    const last = actions[actions.length - 1]
    if (fs.existsSync(last.dest)) errors.push(`${at}: 目标已存在 ${path.relative(REPO, last.dest)}`)
  }
  return { actions, errors }
}

function main() {
  const argv = process.argv.slice(2)
  const apply = argv.includes('--apply')
  const reportFile = argv.find((a) => !a.startsWith('--'))
  if (!reportFile || !fs.existsSync(reportFile)) {
    console.error('用法:npm run papers:file -- <扫描报告路径> [--apply]')
    process.exit(1)
  }

  const { actions, errors } = plan(parseReport(reportFile))
  if (errors.length > 0) {
    console.error('报告有问题,一个文件都没动:')
    for (const e of errors) console.error(`- ${e}`)
    process.exit(1)
  }
  if (actions.length === 0) {
    console.log('没有需要归档的行(去向都是「跳过」或「待定」)。')
    return
  }

  for (const a of actions) {
    console.log(`[${a.kind}] ${path.basename(a.source)}\n        → ${a.dest.replace(os.homedir(), '~')}`)
    if (a.row) console.log(`        索引 ## ${a.topic}  ${a.row}`)
  }
  if (!apply) {
    console.log(`\n以上是干跑,${actions.length} 项都没执行。确认无误后加 --apply。`)
    return
  }

  for (const a of actions) {
    fs.mkdirSync(path.dirname(a.dest), { recursive: true })
    fs.renameSync(a.source, a.dest) // 只移动,绝不删除
    if (a.row) register(a.index, a.topic, a.row)
    if (a.verdict) appendVerdict(a.verdict)
  }
  console.log(`\n已归档 ${actions.length} 项。`)
  console.log('不收的原件在废纸篓:', TRASH.replace(os.homedir(), '~'))
  console.log('下一步:npm run reports:status 与 npm run readings:status 看告警,再 npm test。')
}

main()
