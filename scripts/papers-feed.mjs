// 材料分流 · 来源二「alphaXiv 热榜」的第 1 步扫描(docs/10-材料解读流程.md 第二节)
//
// 拉 alphaXiv 首页热榜(公开接口,不用登录),挑出首发 7 天内票数已达线、属于计算机方向、
// 还没记过的论文,按 readings/_机构名单.md 给出初判。只采集与初判,**不下载、不登记、不移动**;
// 核实、取原件、登记索引与解读由维护者或每日定时任务接着做。
//
// 用法:
//   npm run papers:feed                        干跑,打印今天新达标的论文
//   npm run papers:feed -- --write             同时把新行追加进 readings/_alphaxiv-扫描记录.md
//   npm run papers:feed -- --threshold 30 --days 7
//   npm run papers:feed -- --input <feed.json> 用本地 JSON 代替联网(测试与排障)
import fs from 'node:fs'
import path from 'node:path'

const REPO = process.cwd()
const FEED_URL = 'https://api.alphaxiv.org/papers/v3/feed'
const SORTS = ['Hot', 'Likes']
const PAGES = 4
const PAGE_SIZE = 50

export const RECORD_FILE = path.join(REPO, 'readings', '_alphaxiv-扫描记录.md')
export const ORG_FILE = path.join(REPO, 'readings', '_机构名单.md')

function parseArgs(argv) {
  const args = { write: false, threshold: 30, days: 7, input: null }
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--write') args.write = true
    else if (argv[i] === '--threshold') args.threshold = Number(argv[++i])
    else if (argv[i] === '--days') args.days = Number(argv[++i])
    else if (argv[i] === '--input') args.input = path.resolve(argv[++i])
  }
  return args
}

export function todayInShanghai(now = new Date()) {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai' }).format(now)
}

function daysBetween(from, to) {
  return Math.round((Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / 86_400_000)
}

/** 机构名单:`## 分组` 下每行 `- 目录名 | 别名 | 别名` */
export function parseOrgList(text) {
  const orgs = []
  let group = null
  for (const line of text.split(/\r?\n/)) {
    const heading = line.match(/^##\s+(.+?)\s*$/)
    if (heading) {
      group = heading[1]
      continue
    }
    const item = line.match(/^-\s+(.+)$/)
    if (!item || !group || !item[1].includes('|')) continue
    const [dir, ...aliases] = item[1].split('|').map((cell) => cell.trim())
    if (dir && aliases.length > 0) orgs.push({ dir, group, aliases: aliases.filter(Boolean) })
  }
  return orgs
}

/** 别名匹配:忽略大小写后相等,或以「别名 + 空格/逗号」开头。目录名本身不参与匹配 */
export function matchOrg(name, orgs) {
  const lower = name.trim().toLowerCase()
  for (const org of orgs) {
    for (const alias of org.aliases) {
      const a = alias.toLowerCase()
      if (lower === a || lower.startsWith(`${a} `) || lower.startsWith(`${a},`)) return org
    }
  }
  return null
}

/** 与 lib/alphaxiv.ts 的同名函数读同一份文件,测试守两边一致 */
export function parseScanRecord(text) {
  const rows = []
  for (const line of text.split(/\r?\n/)) {
    if (!line.startsWith('|')) continue
    const cells = line.split('|').slice(1, -1).map((cell) => cell.trim())
    if (cells.length < 7 || !/^\d{4}-\d{2}-\d{2}$/.test(cells[0])) continue
    const [date, id, title, orgs, votes, verdict, destination] = cells
    rows.push({ date, id, title, orgs, votes: Number(votes), verdict, destination })
  }
  return rows
}

const normalize = (text) => text.toLowerCase().replace(/[^a-z0-9一-鿿]/g, '')

/** 两个索引与判定档案里登记过的材料名;标题冒号前的部分与之相同,就当已在库 */
export function knownSlugs(repo = REPO) {
  const slugs = new Set()
  const files = [
    path.join(repo, 'reports', 'index.md'),
    path.join(repo, 'readings', 'index.md'),
    path.join(repo, 'readings', '_判定档案.md'),
  ]
  for (const file of files) {
    if (!fs.existsSync(file)) continue
    for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
      if (!line.startsWith('|')) continue
      const first = line.split('|')[1]?.trim().replace(/^`|`$/g, '')
      if (first && !/^[-:]+$/.test(first)) slugs.add(normalize(first))
    }
  }
  slugs.delete('')
  return slugs
}

const isComputerScience = (topics) =>
  topics.some((t) => t === 'Computer Science' || t.startsWith('cs.') || t === 'stat.ML')

const cell = (text) => String(text).replace(/\|/g, '/').replace(/\s+/g, ' ').trim()

/**
 * 纯函数:从热榜论文里挑出今天要新记的行。
 * 条件:票数 ≥ 线、首发距今 ≤ days、计算机方向、编号不在扫描记录里。
 */
export function selectCandidates(papers, { today, threshold, days, recordIds, orgs, known }) {
  const seen = new Set()
  const rows = []
  for (const paper of papers) {
    const id = paper.universal_paper_id
    if (!id || seen.has(id) || recordIds.has(id)) continue
    seen.add(id)
    const votes = paper.metrics?.public_total_votes ?? 0
    const published = String(paper.first_publication_date ?? '').slice(0, 10)
    if (votes < threshold || !published) continue
    const age = daysBetween(published, today)
    if (age < 0 || age > days) continue
    if (!isComputerScience(paper.topics ?? [])) continue

    const orgNames = (paper.organization_info ?? []).map((o) => o.name).filter(Boolean)
    const matched = orgNames.map((name) => matchOrg(name, orgs)).find(Boolean) ?? null
    const head = normalize(String(paper.title).split(':')[0])
    const verdict = known.has(head)
      ? 'D · 已在库'
      : matched
        ? `A · 名单内(${matched.dir})`
        : '待核 · 名单外'
    rows.push({
      date: today,
      id,
      title: cell(paper.title),
      orgs: cell(orgNames.slice(0, 4).join('、') || '(alphaXiv 未标)'),
      votes,
      verdict,
      destination: '待定',
      published,
      url: `https://www.alphaxiv.org/abs/${id}`,
    })
  }
  return rows.sort((a, b) => b.votes - a.votes)
}

/** 在表格最后一行后面追加;扫描记录里不存在表格时报错,不自作主张重建文件 */
export function appendRows(text, rows) {
  const lines = text.replace(/\n+$/, '').split('\n')
  let last = -1
  lines.forEach((line, index) => {
    if (line.startsWith('|')) last = index
  })
  if (last === -1) throw new Error('扫描记录里找不到表格')
  const added = rows.map(
    (r) => `| ${r.date} | ${r.id} | ${r.title} | ${r.orgs} | ${r.votes} | ${r.verdict} | ${r.destination} |`,
  )
  lines.splice(last + 1, 0, ...added)
  return lines.join('\n') + '\n'
}

async function fetchFeed() {
  const papers = []
  for (const sort of SORTS) {
    for (let page = 0; page < PAGES; page++) {
      const query = new URLSearchParams({
        pageNum: String(page),
        sort,
        pageSize: String(PAGE_SIZE),
        interval: '7 Days',
        topics: '[]',
      })
      const res = await fetch(`${FEED_URL}?${query}`, { headers: { accept: 'application/json' } })
      if (!res.ok) {
        throw new Error(`alphaXiv 热榜接口返回 ${res.status}:${(await res.text()).slice(0, 300)}`)
      }
      const body = await res.json()
      if (!Array.isArray(body.papers)) throw new Error('alphaXiv 热榜接口的返回里没有 papers 数组,接口可能改版')
      papers.push(...body.papers)
      if (body.papers.length < PAGE_SIZE) break
    }
  }
  return papers
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const papers = args.input ? JSON.parse(fs.readFileSync(args.input, 'utf8')).papers : await fetchFeed()
  const recordText = fs.readFileSync(RECORD_FILE, 'utf8')
  const rows = selectCandidates(papers, {
    today: todayInShanghai(),
    threshold: args.threshold,
    days: args.days,
    recordIds: new Set(parseScanRecord(recordText).map((r) => r.id)),
    orgs: parseOrgList(fs.readFileSync(ORG_FILE, 'utf8')),
    known: knownSlugs(),
  })

  console.log(`热榜共 ${papers.length} 条(含两种排序的重复);今天新达标 ${rows.length} 篇(≥${args.threshold} 票、首发 ${args.days} 天内、计算机方向)\n`)
  for (const r of rows) {
    console.log(`${String(r.votes).padStart(4)} 票  ${r.published}  ${r.verdict}`)
    console.log(`      ${r.title}`)
    console.log(`      ${r.orgs}`)
    console.log(`      ${r.url}\n`)
  }
  if (args.write && rows.length > 0) {
    fs.writeFileSync(RECORD_FILE, appendRows(recordText, rows), 'utf8')
    console.log(`已追加 ${rows.length} 行到 ${path.relative(REPO, RECORD_FILE)}`)
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error.message)
    process.exit(1)
  })
}
