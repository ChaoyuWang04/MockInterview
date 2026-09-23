// 把云端每日任务入库的原件补到本机(docs/10-材料解读流程.md 第二节「来源二」)
//
// PDF 原件不进 git,云端只推送解读与扫描记录。本脚本按 readings/_alphaxiv-扫描记录.md 的「去向」
// 找出本机缺的原件下载:arXiv 编号走 arxiv.org,非 arXiv 的(alphaXiv 自有编号,如厂商报告)走 alphaXiv 托管的 PDF。
//
// 用法:
//   npm run papers:pull            干跑,列出缺哪些
//   npm run papers:pull -- --apply 真的下载
import fs from 'node:fs'
import path from 'node:path'

import { RECORD_FILE, parseScanRecord } from './papers-feed.mjs'

const REPO = process.cwd()
const ARXIV_ID = /^\d{4}\.\d{4,5}$/

/** 去向 → 本机原件路径;不是入库去向时返回 null */
export function sourcePathOf(destination, repo = REPO) {
  const clean = destination.replace(/^`|`$/g, '')
  const match = clean.match(/^(reports|readings)\/([^/]+)\/([^/]+)$/)
  if (!match) return null
  const [, library, dir, slug] = match
  return library === 'reports'
    ? path.join(repo, 'papers', dir, `${slug}.pdf`)
    : path.join(repo, 'readings', '_src', dir, `${slug}.pdf`)
}

export function missingSources(rows, repo = REPO) {
  return rows
    .filter((row) => !row.verdict.startsWith('D'))
    .map((row) => ({ ...row, file: sourcePathOf(row.destination, repo) }))
    .filter((row) => row.file && !fs.existsSync(row.file))
}

async function main() {
  const apply = process.argv.includes('--apply')
  const missing = missingSources(parseScanRecord(fs.readFileSync(RECORD_FILE, 'utf8')))
  if (missing.length === 0) {
    console.log('扫描记录里入库的原件本机都有了')
    return
  }
  let failed = 0
  for (const row of missing) {
    const rel = path.relative(REPO, row.file)
    const url = ARXIV_ID.test(row.id)
      ? `https://arxiv.org/pdf/${row.id}`
      : `https://www.alphaxiv.org/abs/${row.id}.pdf`
    if (!apply) {
      console.log(`待下载    ${rel}  ← ${url}`)
      continue
    }
    // 单个请求限时,一个连接挂住不拖死整批;失败的记下来,下次再跑会重试
    let res
    let bytes
    try {
      res = await fetch(url, { signal: AbortSignal.timeout(90_000) })
      bytes = Buffer.from(await res.arrayBuffer())
    } catch (error) {
      console.log(`下载失败  ${rel}  ← ${url}(${error.name === 'TimeoutError' ? '90 秒超时' : error.message})`)
      failed++
      continue
    }
    if (!res.ok || bytes.subarray(0, 4).toString() !== '%PDF') {
      console.log(`下载失败  ${rel}  ← ${url}(HTTP ${res.status})`)
      failed++
      continue
    }
    fs.mkdirSync(path.dirname(row.file), { recursive: true })
    fs.writeFileSync(row.file, bytes)
    console.log(`已下载    ${rel}  (${(bytes.length / 1e6).toFixed(1)} MB)`)
  }
  if (!apply) console.log('\n加 -- --apply 真的下载')
  if (failed > 0) process.exit(1)
}

if (import.meta.url === `file://${process.argv[1]}`) main()
