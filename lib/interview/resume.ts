import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'
import type { SamplerProfile } from './sampler'

export interface ResumeEntry {
  /** 文件名去掉 .md,如 `02-infra`;是简历的稳定 id */
  slug: string
  /** 选择器上显示的名字,如 `Infra 档` */
  name: string
  role: string
  /** 在册章节 —— 抽题的硬门禁,人工维护 */
  chapters: string[]
  /** 章节权重 —— 这场面试各方向的出题比例,同样人工维护 */
  chapterWeights: Record<string, number>
  isDefault: boolean
  /** 正文(不含 frontmatter),画像脚本的输入 */
  body: string
}

/**
 * `章节:` 两种写法都认:
 *   `章节: [03-强化学习, 04-Infra]`              → 各章等权(都算 1)
 *   `章节: {03-强化学习: 3, 04-Infra: 1}`        → 强化学习出题量是 Infra 的三倍
 *
 * 权重 ≤ 0 视为不在册(和没写这一章等价)—— 这样临时关掉一个方向只要改个数字。
 */
export function parseChapters(raw: unknown): {
  chapters: string[]
  chapterWeights: Record<string, number>
} {
  if (Array.isArray(raw)) {
    const chapters = raw.map(String)
    return { chapters, chapterWeights: Object.fromEntries(chapters.map((c) => [c, 1])) }
  }
  if (raw && typeof raw === 'object') {
    const weights: Record<string, number> = {}
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      const n = Number(v)
      if (Number.isFinite(n) && n > 0) weights[k] = n
    }
    return { chapters: Object.keys(weights), chapterWeights: weights }
  }
  return { chapters: [], chapterWeights: {} }
}

export function resumesRoot(): string {
  return path.join(process.cwd(), 'interview', 'resumes')
}

export function profilesRoot(): string {
  return path.join(process.cwd(), 'interview', 'profiles')
}

export function listResumes(root = resumesRoot()): ResumeEntry[] {
  if (!fs.existsSync(root)) return []
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isFile() && d.name.endsWith('.md') && !d.name.startsWith('_') && !d.name.startsWith('.'))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
    .map((file) => {
      const parsed = matter(fs.readFileSync(path.join(root, file), 'utf8'))
      const data = parsed.data as Record<string, unknown>
      const { chapters, chapterWeights } = parseChapters(data['章节'])
      return {
        slug: file.slice(0, -3),
        name: typeof data.name === 'string' ? data.name : file.slice(0, -3),
        role: typeof data.role === 'string' ? data.role : '',
        chapters,
        chapterWeights,
        isDefault: data.default === true,
        body: parsed.content.trim(),
      }
    })
}

export function defaultResume(root = resumesRoot()): ResumeEntry | null {
  const all = listResumes(root)
  return all.find((r) => r.isDefault) ?? all[0] ?? null
}

/** 画像文件的形状;`章节` 不在里面——它只从简历 frontmatter 来,不给 LLM 改的机会 */
export interface ProfileFile {
  resume: string
  resumeHash: string
  topic亲和: Record<string, number>
  项目深挖点: { 项目: string; 可问: string; 关联topic: string[] }[]
  摘要: string
}

export function loadProfile(slug: string, root = profilesRoot()): ProfileFile | null {
  const file = path.join(root, `${slug}.json`)
  if (!fs.existsSync(file)) return null
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8')) as ProfileFile
  } catch {
    return null
  }
}

/**
 * 简历 + 画像 → 抽题器要的 profile。
 * **章节永远取自简历**;画像只贡献软权重。画像缺失时全部按亲和度 1 走,
 * 系统照常可用(只是没有简历加权),不会因为没跑过 LLM 就瘫掉。
 */
export function samplerProfileFor(resume: ResumeEntry, profile: ProfileFile | null): SamplerProfile {
  return {
    chapters: resume.chapters,
    chapterWeights: resume.chapterWeights,
    affinity: profile?.topic亲和 ?? {},
    fallbackAffinity: 1,
  }
}
