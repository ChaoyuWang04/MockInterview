import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

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
  /** `单篇 / CudaGraph`;整场模拟面试时为 undefined */
  scope?: string
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
 * 从复盘正文里摘出模型写的主题行,并把它从正文里剥掉。
 *
 * 复盘是自由 markdown,**不能用 JSON 模式兜底**,所以匹配写得很松:
 * 容全角冒号、前导空白、`**主题**` 这种加粗写法。模型没给就返回 null,
 * 由调用方回落到「覆盖」的文章名 —— 宁可标题难看,也不能让一场记录写不进去。
 */
export function extractTheme(review: string): { theme: string | null; body: string } {
  // 冒号后**只允许同行空白**(`[ \t]*`,不是 `\s*`)—— 用 `\s*` 会跨行,
  // 模型写了空的「主题:」时会把下一行的 `## 整体` 当成主题。测试守着这条。
  const m = review.match(/^[ \t]*(?:\*\*)?主题(?:\*\*)?[ \t]*[::][ \t]*(.*)$/m)
  if (!m) return { theme: null, body: review }
  const theme = m[1].replace(/\*\*/g, '').trim().slice(0, 40)
  if (!theme) return { theme: null, body: review }
  return { theme, body: review.replace(m[0], '').replace(/^\s*\n/, '').trimStart() }
}

/**
 * 写盘。**每轮都整文件重写**——转录很小(一场几十 KB),
 * 整写比增量追加简单得多,也不会因为中途崩溃留下半截结构。
 *
 * ⚠️ 这里用 gray-matter 整体序列化 frontmatter 是**安全的**,和铁律 2 不冲突:
 * 那条禁令针对 `questions/` —— 那是你手写的文件,有行内注释和手写格式要保护。
 * 会话文件是程序生成、每次整写的,没有任何手写内容会被冲掉。
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

  const { theme, body: reviewBody } = extractTheme(review ?? '')
  const modeLabel = meta.mode === 'interview' ? '面试档' : '过题档'

  const body = [
    `# 模拟面试 ${meta.id}`,
    '',
    `- 开始:${meta.startedAt}`,
    `- 简历:${meta.resume} · 模式:${modeLabel}`,
    `- 轮次:${turns.length} · 要点命中:${totalHit}/${totalPoints}`,
    `- 覆盖:${articles.join(' · ') || '—'}`,
    '',
    reviewBody ? `## 复盘\n\n${reviewBody}\n` : '',
    '## 逐轮记录',
    '',
    ...turns.map(fmtTurn),
  ]
    .filter((x) => x !== '')
    .join('\n')

  const front = {
    // 模型没给主题就回落到覆盖的前两篇 —— 列表页至少有个能认的标题
    主题: theme || articles.slice(0, 2).join(' · ') || '未命名',
    简历: meta.resume,
    模式: modeLabel,
    ...(meta.scope ? { 范围: meta.scope } : {}),
    开始: meta.startedAt,
    轮次: turns.length,
    命中: `${totalHit}/${totalPoints}`,
    覆盖: articles,
  }
  fs.writeFileSync(file, matter.stringify(`${body}\n`, front), 'utf8')
  return file
}

/** 列表页要显示的一场 */
export interface SessionSummary {
  id: string
  /** `2026-09-03`,列表按它分组 */
  date: string
  /** `23:37` */
  time: string
  主题: string
  /** `单篇 / CudaGraph`;整场模拟面试时为空 */
  范围: string
  简历: string
  模式: string
  轮次: number
  hit: number
  points: number
  覆盖: string[]
}

const str = (v: unknown, fallback = '') => (typeof v === 'string' ? v : fallback)

/**
 * 读一场并解析出摘要。
 *
 * **对没有 frontmatter 的旧文件要能兜住** —— 第一场真实面试就是那个格式,
 * 而且它还没被人核过判分,不该为了适配新页面去改它。所以 frontmatter 缺什么,
 * 就从正文那几行 `- 开始:` / `- 简历:` 里回落解析。
 */
export function readSession(id: string): { summary: SessionSummary; body: string } | null {
  const file = path.join(sessionsDir(), `${id}.md`)
  if (!fs.existsSync(file)) return null
  const raw = fs.readFileSync(file, 'utf8')
  const { data, content } = matter(raw)

  const line = (label: string) =>
    content.match(new RegExp(`^- ${label}[::]\\s*(.+)$`, 'm'))?.[1]?.trim() ?? ''
  // `- 简历:Agent 档 · 模式:过题档` —— 一行两个字段
  const metaLine = line('简历')
  const 简历 = str(data.简历) || metaLine.split('·')[0]?.replace(/^简历[::]?/, '').trim() || '—'
  const 模式 = str(data.模式) || metaLine.match(/模式[::]\s*(\S+)/)?.[1] || '—'
  const countLine = line('轮次')
  const 轮次 = typeof data.轮次 === 'number' ? data.轮次 : Number(countLine.match(/^(\d+)/)?.[1] ?? 0)
  const hitRaw = str(data.命中) || countLine.match(/命中[::]\s*(\d+\/\d+)/)?.[1] || '0/0'
  const [hit, points] = hitRaw.split('/').map((n) => Number(n) || 0)
  const 覆盖 = Array.isArray(data.覆盖)
    ? (data.覆盖 as unknown[]).map((x) => String(x))
    : line('覆盖').split('·').map((s) => s.trim()).filter((s) => s && s !== '—')

  // id 形如 2026-09-02-2337-agent-rl
  const m = id.match(/^(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})-/)
  return {
    summary: {
      id,
      date: m?.[1] ?? '未知日期',
      time: m ? `${m[2]}:${m[3]}` : '',
      主题: str(data.主题) || 覆盖.slice(0, 2).join(' · ') || '未命名',
      范围: str(data.范围),
      简历,
      模式,
      轮次,
      hit,
      points,
      覆盖,
    },
    body: content.trim(),
  }
}

/** 全部会话摘要,新的在前。目录里几百个文件也就几毫秒,不需要索引。 */
export function listSessionSummaries(): SessionSummary[] {
  return listSessions()
    .map((s) => readSession(s.id)?.summary)
    .filter((s): s is SessionSummary => s !== undefined)
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
