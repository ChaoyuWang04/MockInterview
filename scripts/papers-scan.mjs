// 桌面材料分流 · 第 1 步「扫描」(docs/10-材料解读流程.md 第二节)
//
// 只采集,不判断,**不移动任何文件**。产出一份扫描报告,人核实并改「去向」列后,
// 交给 scripts/papers-file.mjs 归档。
//
// 用法:
//   npm run papers:scan                 扫 ~/Desktop 顶层
//   npm run papers:scan -- --dir <路径>  一次性处理某个目录(递归)
//   npm run papers:scan -- --out <路径>  指定报告输出位置
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const REPO = process.cwd()
const READABLE = /\.(pdf|srt|vtt|txt|md)$/i

const TIER = {
  A: 'A · 建议进报告解读',
  B: 'B · 建议进日常研读',
  C: 'C · ⚠️ 闸门三未过,请逐条拍板',
  D: 'D · 已在库或已判定,跳过',
  E: 'E · 非文献,默认不收,原地不动',
}

function parseArgs(argv) {
  const args = { dir: path.join(os.homedir(), 'Desktop'), recursive: false, out: null }
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--dir') {
      args.dir = path.resolve(argv[++i])
      args.recursive = true
    } else if (argv[i] === '--out') args.out = path.resolve(argv[++i])
  }
  return args
}

function listCandidates(dir, recursive) {
  const out = []
  const walk = (current, depth) => {
    let entries
    try {
      entries = fs.readdirSync(current, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue
      const full = path.join(current, entry.name)
      if (entry.isDirectory()) {
        if (recursive && depth < 4) walk(full, depth + 1)
      } else if (READABLE.test(entry.name)) out.push(full)
    }
  }
  walk(dir, 0)
  return out.sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

function run(cmd, cmdArgs) {
  try {
    return execFileSync(cmd, cmdArgs, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 })
  } catch {
    return ''
  }
}

/** PDF 首页文本 + 页数;非 PDF 直接读前若干字符 */
function extract(file) {
  if (/\.pdf$/i.test(file)) {
    const info = run('pdfinfo', [file])
    const pages = Number(info.match(/^Pages:\s+(\d+)$/m)?.[1] ?? 0)
    const metaTitle = info.match(/^Title:\s+(.+)$/m)?.[1]?.trim() ?? ''
    const head = run('pdftotext', ['-f', '1', '-l', '2', '-q', file, '-'])
    return { pages, metaTitle, head, broken: pages === 0 }
  }
  let head = ''
  try {
    head = fs.readFileSync(file, 'utf8').slice(0, 8000)
  } catch {
    return { pages: 0, metaTitle: '', head: '', broken: true }
  }
  return { pages: 0, metaTitle: '', head, broken: false }
}

/** 像不像一份可解读的材料。拿不准的一律进 E 让人看,不自作主张丢掉 */
function looksLikeMaterial(file, { head, broken }) {
  if (broken) return { ok: false, why: '文件读不出页数,可能损坏或不是 PDF' }
  const text = head.toLowerCase()
  const hits = []
  if (/\babstract\b|\b摘\s*要\b/.test(text)) hits.push('有摘要')
  if (/arxiv[:\s]*\d{4}\.\d{4,5}/i.test(head)) hits.push('有 arXiv 编号')
  if (/\buniversity\b|\binstitute\b|\blaborator/.test(text)) hits.push('有机构署名')
  if (/\bwe (propose|present|introduce|show)\b/.test(text)) hits.push('有论文式表述')
  if (/^\d{4}-/.test(path.basename(file))) hits.push('文件名带年份前缀')
  if (hits.length === 0) {
    return {
      ok: false,
      why: isTranscript(file)
        ? '转录文本,非文献,默认不收——要收请直接点名'
        : '非文献:首页没有摘要、机构或 arXiv 迹象',
    }
  }
  return { ok: true, why: hits.join('、') }
}

/** 认出转录文本只为把「为什么不收」写准。**扫描器只提名文献**;
 *  网页、字幕、视频这些载体能进库,但要由维护者点名,不由扫描自动提议 */
function isSurvey(file, head) {
  return /\bsurvey\b|\breview\b|综述/i.test(path.basename(file)) ||
    /\b(this|our|a comprehensive) survey\b/i.test(head)
}

function isTranscript(file) {
  const name = path.basename(file)
  return /\.(srt|vtt)$/i.test(name) || /字幕|转录|转写|录音|访谈|交流会|分享会|发布会|实录|transcript|subtitle/i.test(name)
}

function detect(head) {
  const arxiv = head.match(/arxiv[:\s]*(\d{4}\.\d{4,5})/i)?.[1] ?? ''
  const repo =
    head.match(/https?:\/\/github\.com\/[\w.-]+\/[\w.-]+/i)?.[0] ??
    head.match(/github\.com\/[\w.-]+\/[\w.-]+/i)?.[0] ??
    ''
  const year = head.match(/\b(20[12]\d)\b/)?.[1] ?? ''
  return { arxiv, repo, year }
}

/** 文件名里的稳定名字:去掉年份前缀与扩展名,作为 slug 的建议值 */
function slugFrom(file) {
  return path
    .basename(file)
    .replace(READABLE, '')
    .replace(/^(19|20)\d{2}[-_ ]+/, '')
    .replace(/\s+/g, ' ')
    .trim()
}

/** 归一化后用于比对三张表:去掉大小写、空格与常见标点 */
const norm = (s) => s.toLowerCase().replace(/[\s\-_·:：,，.。、"'"'()（）[\]]/g, '')

function readKnown() {
  const known = new Map()
  const add = (name, where) => {
    if (name) known.set(norm(name), where)
  }
  const rows = (file, where) => {
    const full = path.join(REPO, file)
    if (!fs.existsSync(full)) return
    for (const line of fs.readFileSync(full, 'utf8').split(/\r?\n/)) {
      if (!line.startsWith('|')) continue
      const cells = line.split('|').slice(1, -1).map((c) => c.trim())
      if (cells.length < 2 || /^:?-+:?$/.test(cells[0])) continue
      add(cells[0].replace(/`/g, ''), where)
    }
  }
  rows('reports/index.md', '报告解读')
  rows('readings/index.md', '日常研读')
  rows('readings/_判定档案.md', '判定档案')
  return known
}

function classify(file, info, known) {
  // 先查去重再判形态:已经收过的东西不该被重新判一遍
  // (Ultra-Scale-Playbook 的本地打印件首页没有摘要,先判形态就会被误扔进 E)
  const slug = slugFrom(file)
  const key = norm(slug)
  const hit = known.get(key)
  if (hit) return { tier: 'D', reason: `已在${hit}`, target: '跳过' }
  // 本地文件名常带语言或版本后缀(ultrascale-playbook-**zh**),精确比对会漏
  for (const [other, where] of known) {
    if (other.length < 8 || key.length < 8) continue
    if (!other.startsWith(key) && !key.startsWith(other)) continue
    return { tier: 'D', reason: `疑似已在${where}(对上「${other}」,请确认)`, target: '跳过' }
  }

  const material = looksLikeMaterial(file, info)
  if (!material.ok) return { tier: 'E', reason: material.why, target: '跳过' }

  const { arxiv, repo } = detect(info.head)
  const gates = []
  if (repo) gates.push(`有仓库 ${repo}`)
  if (arxiv) gates.push(`arXiv ${arxiv}`)
  // 综述免闸门三:新领域的综述往往出自小机构,拿机构与顶会卡它会把整个方向挡在外面
  if (isSurvey(file, info.head)) {
    return { tier: 'B', reason: `综述,免闸门三;写法见手册第四节${gates.length ? ';' + gates.join('、') : ''}`, target: '待定' }
  }
  // 闸门三的机构与顶会两条脚本判不了,留给核实那一步
  if (gates.length === 0) {
    return {
      tier: 'C',
      reason: '首页没找到开源仓库,机构与顶会录用需人工核实',
      target: '待定',
    }
  }
  return { tier: 'B', reason: `${gates.join('、')};归属与重要程度待核实`, target: '待定' }
}

function report(rows, args) {
  const today = new Date().toISOString().slice(0, 10)
  const out = [
    `# 扫描报告 ${today}`,
    '',
    `扫描范围:\`${args.dir}\`${args.recursive ? '(递归)' : '(仅顶层)'}`,
    '',
    '**这份报告是临时产物,不入库。** 脚本只采集不判断:A/B 是建议,不是判决。',
    '',
    '**扫描只提名文献。** 网页、博客、字幕、录音转写这些载体照样能进库(手册「三种原件形态」),',
    '但要由你点名,不由扫描自动提议——所以它们都在 E 档,附上认出来的类型,方便你挑。',
    '核实后改「去向」列,再跑 `npm run papers:file -- <本文件>` 归档。去向的写法:',
    '',
    '| 去向的写法 | 含义 |',
    '|---|---|',
    '| `报告/<方向>/<公司>` | 移进 `papers/<公司>/`,在 `reports/index.md` 的该方向登记 |',
    '| `研读/<方向>` | 移进 `readings/_src/<方向>/`,在 `readings/index.md` 的该方向登记 |',
    '| `不收:<理由>` | 移进废纸篓,在 `readings/_判定档案.md` 记一行。**理由必写**,否则下轮会重新调研一遍 |',
    '| `跳过` | 什么都不做,原地留着 |',
    '',
    '进两个库的条目**必须填「一句话」**,那是索引行的第三列;留空归档器会拒绝。',
    '「建议名」是从文件名推的 slug,可以直接改。',
    '',
  ]
  for (const [tier, title] of Object.entries(TIER)) {
    const group = rows.filter((row) => row.tier === tier)
    out.push(`## ${title}  (${group.length})`, '')
    if (group.length === 0) {
      out.push('无。', '')
      continue
    }
    out.push('| 文件 | 建议名 | 去向 | 一句话 | 依据 |', '|---|---|---|---|---|')
    for (const row of group) {
      out.push(
        `| \`${row.file}\` | ${row.slug} | ${row.target} |  | ${row.reason.replace(/\|/g, '\\|')} |`,
      )
    }
    out.push('')
  }
  return out.join('\n')
}

function main() {
  const args = parseArgs(process.argv.slice(2))
  if (!fs.existsSync(args.dir)) {
    console.error(`扫描目录不存在:${args.dir}`)
    process.exit(1)
  }
  const known = readKnown()
  const files = listCandidates(args.dir, args.recursive)
  const rows = files.map((file) => {
    const info = extract(file)
    return { file, slug: slugFrom(file), ...classify(file, info, known) }
  })

  const out = args.out ?? path.join(REPO, `_扫描报告-${new Date().toISOString().slice(0, 10)}.md`)
  fs.writeFileSync(out, report(rows, args))

  const count = (tier) => rows.filter((row) => row.tier === tier).length
  console.log(`扫了 ${files.length} 个文件:${args.dir}`)
  for (const [tier, title] of Object.entries(TIER)) console.log(`  ${title}  ${count(tier)}`)
  console.log(`\n报告写到 ${path.relative(REPO, out)}`)
  console.log('下一步:核实并改「去向」列,然后 npm run papers:file -- <报告>')
}

main()
