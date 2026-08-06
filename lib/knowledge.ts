import fs from 'node:fs'
import path from 'node:path'

export function knowledgeRoot(): string {
  return path.join(process.cwd(), 'knowledge')
}

function isVisible(name: string): boolean {
  return !name.startsWith('.') && !name.startsWith('_')
}

export function listKbCategories(root = knowledgeRoot()): string[] {
  if (!fs.existsSync(root)) return []
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory() && isVisible(d.name))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 某分类下的文章主题名列表(文件名去掉 .md,与题目 frontmatter topic 第一段一一对应) */
export function listArticleTopics(category: string, root = knowledgeRoot()): string[] {
  const dir = path.join(root, category)
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isFile() && d.name.endsWith('.md') && isVisible(d.name))
    .map((d) => d.name.slice(0, -3))
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 白名单校验后返回文章 markdown 原文;不存在返回 null */
export function getArticle(category: string, topic: string, root = knowledgeRoot()): string | null {
  if (!listKbCategories(root).includes(category)) return null
  if (!listArticleTopics(category, root).includes(topic)) return null
  return fs.readFileSync(path.join(root, category, `${topic}.md`), 'utf8')
}
