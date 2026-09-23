import fs from 'node:fs'
import path from 'node:path'

export interface OsPage {
  file: string
  title: string
  content: string
}

export function opensourceRoot(): string {
  return path.join(process.cwd(), 'opensource')
}

function isVisible(name: string): boolean {
  return !name.startsWith('.') && !name.startsWith('_')
}

function listDirs(dir: string): string[] {
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && isVisible(d.name))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 主页分组顺序:从上层服务到底层硬件;不在表里的主题按字母序排在最后 */
export const OS_TOPIC_ORDER = ['推理服务', '分布式训练', 'RL训练框架', '多模态', '框架内核', '通信', 'Kernel与GPU编程']

/** 组内顺序:多模态按「推理 → 训练 → RL」排;没列的主题与项目按字母序 */
export const OS_PROJECT_ORDER: Record<string, string[]> = {
  多模态: ['vllm-omni', 'sglang-omni', 'VeOmni', 'verl-omni'],
}

function byOrder(names: string[], order: string[] = []): string[] {
  const rank = (n: string) => (order.includes(n) ? order.indexOf(n) : order.length)
  return [...names].sort((a, b) => rank(a) - rank(b))
}

export function listOsTopics(root = opensourceRoot()): string[] {
  return byOrder(listDirs(root), OS_TOPIC_ORDER)
}

export function listOsProjects(topic: string, root = opensourceRoot()): string[] {
  return byOrder(listDirs(path.join(root, topic)), OS_PROJECT_ORDER[topic])
}

/** 白名单校验:topic/project 必须真实存在于扫描结果中 */
export function isValidOsProject(topic: string, project: string, root = opensourceRoot()): boolean {
  return listOsTopics(root).includes(topic) && listOsProjects(topic, root).includes(project)
}

/** 项目的全部解读页,按文件名(NN- 前缀)排序;标题 = 去掉序号前缀与扩展名 */
export function getOsPages(topic: string, project: string, root = opensourceRoot()): OsPage[] {
  if (!isValidOsProject(topic, project, root)) return []
  const dir = path.join(root, topic, project)
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isFile() && d.name.endsWith('.md') && isVisible(d.name))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
    .map((file) => ({
      file,
      title: file.replace(/^\d+-/, '').replace(/\.md$/, ''),
      content: fs.readFileSync(path.join(dir, file), 'utf8'),
    }))
}

/** 仅统计页数(索引页用,不读内容) */
export function countOsPages(topic: string, project: string, root = opensourceRoot()): number {
  if (!isValidOsProject(topic, project, root)) return 0
  return fs.readdirSync(path.join(root, topic, project)).filter((f) => f.endsWith('.md') && isVisible(f)).length
}
