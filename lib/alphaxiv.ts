/**
 * alphaXiv 扫描记录的页面侧读取:只取「去向 → 票数」,给两个解读库的「按时间」视图排序。
 *
 * 写入与分流在 scripts/papers-feed.mjs;这里的解析与脚本里的 parseScanRecord 读同一份文件,
 * tests/alphaxiv.test.ts 守两边一致。规则见 docs/10-材料解读流程.md 第二节「来源二」。
 */
import fs from 'node:fs'
import path from 'node:path'

export interface ScanRecordRow {
  date: string
  id: string
  title: string
  orgs: string
  votes: number
  verdict: string
  destination: string
}

export function scanRecordPath(repo = process.cwd()): string {
  return path.join(repo, 'readings', '_alphaxiv-扫描记录.md')
}

export function parseScanRecord(text: string): ScanRecordRow[] {
  const rows: ScanRecordRow[] = []
  for (const line of text.split(/\r?\n/)) {
    if (!line.startsWith('|')) continue
    const cells = line.split('|').slice(1, -1).map((cell) => cell.trim())
    if (cells.length < 7 || !/^\d{4}-\d{2}-\d{2}$/.test(cells[0])) continue
    const [date, id, title, orgs, votes, verdict, destination] = cells
    rows.push({ date, id, title, orgs, votes: Number(votes), verdict, destination })
  }
  return rows
}

/**
 * 某个库(`reports` 或 `readings`)里「<目录>/<材料>」→ 达标当天票数。
 * 同一篇若被记了两次(不该发生),取先记的那次。
 */
export function votesByArticle(
  library: 'reports' | 'readings',
  repo = process.cwd(),
): Map<string, number> {
  const file = scanRecordPath(repo)
  const votes = new Map<string, number>()
  if (!fs.existsSync(file)) return votes
  const prefix = `${library}/`
  for (const row of parseScanRecord(fs.readFileSync(file, 'utf8'))) {
    const destination = row.destination.replace(/^`|`$/g, '')
    if (!destination.startsWith(prefix) || !Number.isFinite(row.votes)) continue
    const key = destination.slice(prefix.length)
    if (!votes.has(key)) votes.set(key, row.votes)
  }
  return votes
}
