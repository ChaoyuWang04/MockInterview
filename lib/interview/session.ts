import fs from 'node:fs'
import path from 'node:path'

/**
 * 会话落盘。
 *
 * 一场面试一个 markdown 文件,和题库/知识库同构 —— 这是仓库第一铁律
 * 「markdown 即数据库」在这个模块的落点:练完留下的东西可以 grep、可以 diff、
 * 可以进 git,而不是关掉标签页就没了。
 */
export interface SessionTurn {
  /** 面试官问出口的话 */
  ask: string
  /** 你的回答(语音转写或打字) */
  answer: string
  hit: number[]
  miss: number[]
  /** 本题的要点原文,用来在复盘里指出「漏了哪条」 */
  points: string[]
  /** 面试官的私下备注 */
  note?: string
  article: string
  questionId: string
  skipped?: boolean
}

export interface SessionMeta {
  id: string
  startedAt: string
  resume: string
  mode: 'interview' | 'drill'
  voice?: string
}

export function sessionsDir(): string {
  return path.join(process.cwd(), 'interview', 'sessions')
}

/** `2026-09-02-2210-infra` —— 排序即时间序,和题库的 NN- 前缀一个思路 */
export function newSessionId(resume: string, at = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0')
  const stamp =
    `${at.getFullYear()}-${p(at.getMonth() + 1)}-${p(at.getDate())}-` +
    `${p(at.getHours())}${p(at.getMinutes())}`
  return `${stamp}-${resume.replace(/^\d+-/, '')}`
}

function fmtTurn(t: SessionTurn, i: number): string {
  const lines = [`### ${i + 1}. ${t.ask}`, '']
  if (t.skipped) {
    lines.push('> 跳过', '')
    return lines.join('\n')
  }
  lines.push(t.answer || '(没有回答)', '')
  if (t.points.length) {
    lines.push(...t.points.map((p, k) => `- ${t.hit.includes(k) ? '✅' : '❌'} ${p}`), '')
  }
  if (t.note) lines.push(`> 面试官备注:${t.note}`, '')
  return lines.join('\n')
}

/**
 * 写盘。**每轮都整文件重写**——转录很小(一场几十 KB),
 * 整写比增量追加简单得多,也不会因为中途崩溃留下半截结构。
 */
export function writeSession(
  meta: SessionMeta,
  turns: SessionTurn[],
  review?: string,
): string {
  const dir = sessionsDir()
  fs.mkdirSync(dir, { recursive: true })
  const file = path.join(dir, `${meta.id}.md`)

  const totalHit = turns.reduce((s, t) => s + t.hit.length, 0)
  const totalPoints = turns.reduce((s, t) => s + t.points.length, 0)
  const articles = [...new Set(turns.map((t) => t.article).filter(Boolean))]

  const body = [
    `# 模拟面试 ${meta.id}`,
    '',
    `- 开始:${meta.startedAt}`,
    `- 简历:${meta.resume} · 模式:${meta.mode === 'interview' ? '面试档' : '过题档'}`,
    `- 轮次:${turns.length} · 要点命中:${totalHit}/${totalPoints}`,
    `- 覆盖:${articles.join(' · ') || '—'}`,
    '',
    review ? `## 复盘\n\n${review}\n` : '',
    '## 逐轮记录',
    '',
    ...turns.map(fmtTurn),
  ]
    .filter((x) => x !== '')
    .join('\n')

  fs.writeFileSync(file, body.endsWith('\n') ? body : `${body}\n`, 'utf8')
  return file
}

export function listSessions(): { id: string; file: string; mtime: number }[] {
  const dir = sessionsDir()
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith('.md'))
    .map((f) => ({
      id: f.slice(0, -3),
      file: path.join(dir, f),
      mtime: fs.statSync(path.join(dir, f)).mtimeMs,
    }))
    .sort((a, b) => b.mtime - a.mtime)
}
