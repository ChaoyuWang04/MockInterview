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
  isDefault: boolean
  /** 正文(不含 frontmatter),画像脚本的输入 */
  body: string
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
      return {
        slug: file.slice(0, -3),
        name: typeof data.name === 'string' ? data.name : file.slice(0, -3),
        role: typeof data.role === 'string' ? data.role : '',
        chapters: Array.isArray(data['章节']) ? data['章节'].map(String) : [],
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
    affinity: profile?.topic亲和 ?? {},
    fallbackAffinity: 1,
  }
}
