/**
 * 已读标记:两个解读库各一份 `<库根>/_已读.md`,每行 `- <目录>/<材料>`。
 *
 * 与 leetcode 的高频标记同一个做法:标记单独存放,不碰索引与正文;勾一次只增或删那一行,
 * 其余行(包括人手写的说明)原样保留。下划线开头,不进网页。
 */
import fs from 'node:fs'
import path from 'node:path'

const HEADER = [
  '# 已读标记',
  '',
  '页面上勾选卡片右下角的「已读」即可增删,程序只增删对应那一行。每行 `- <目录>/<材料>`,顺序不重要。',
  '',
]

const LINE = /^-\s+(.+?)\s*$/

export function readMarksPath(root: string): string {
  return path.join(root, '_已读.md')
}

export function listReadMarks(root: string): Set<string> {
  const file = readMarksPath(root)
  if (!fs.existsSync(file)) return new Set()
  return new Set(
    fs
      .readFileSync(file, 'utf8')
      .split(/\r?\n/)
      .map((line) => line.match(LINE)?.[1])
      .filter((key): key is string => !!key),
  )
}

export function setReadMark(root: string, key: string, read: boolean): void {
  const file = readMarksPath(root)
  const lines = fs.existsSync(file)
    ? fs.readFileSync(file, 'utf8').replace(/\n+$/, '').split(/\r?\n/)
    : [...HEADER]
  const at = lines.findIndex((line) => line.match(LINE)?.[1] === key)
  if (read && at === -1) lines.push(`- ${key}`)
  else if (!read && at !== -1) lines.splice(at, 1)
  else return
  const tmp = `${file}.${process.pid}.tmp`
  fs.writeFileSync(tmp, lines.join('\n') + '\n', 'utf8')
  fs.renameSync(tmp, file)
}
