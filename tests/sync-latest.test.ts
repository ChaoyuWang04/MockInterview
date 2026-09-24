import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const projectRoot = path.resolve(import.meta.dirname, '..')
const syncScript = path.join(projectRoot, 'scripts/sync-latest.sh')

/** 临时仓库:main 分支、一次初始提交,没有 origin,所以脚本在 fetch 处停下 */
function tempRepo(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sync-latest-test-'))
  const git = (...args: string[]) => execFileSync('git', args, { cwd: dir, stdio: 'ignore' })
  git('init', '-q', '-b', 'main')
  git('config', 'user.email', 'test@example.com')
  git('config', 'user.name', 'test')
  fs.mkdirSync(path.join(dir, 'scripts'))
  fs.copyFileSync(syncScript, path.join(dir, 'scripts/sync-latest.sh'))
  git('add', '.')
  git('commit', '-q', '-m', 'init')
  return dir
}

function run(dir: string): string {
  return execFileSync('bash', [path.join(dir, 'scripts/sync-latest.sh')], { encoding: 'utf8' })
}

describe('会话启动同步的未收尾提醒', () => {
  it('内容目录有没提交的文件时按目录计数报出,路径带中文和空格也不乱', () => {
    const dir = tempRepo()
    fs.mkdirSync(path.join(dir, 'opensource/框架内核'), { recursive: true })
    fs.writeFileSync(path.join(dir, 'opensource/框架内核/_临时 草稿.md'), 'x')
    fs.mkdirSync(path.join(dir, 'reports/Ai2'), { recursive: true })
    fs.writeFileSync(path.join(dir, 'reports/Ai2/Olmo-3.md'), 'x')

    const out = run(dir)
    expect(out).toContain('未收尾:内容目录有 2 个文件改了没提交(opensource 1、reports 1)')
  })

  it('内容目录干净、只有代码改动时不提醒', () => {
    const dir = tempRepo()
    fs.writeFileSync(path.join(dir, 'scripts/other.sh'), 'x')

    expect(run(dir)).not.toContain('未收尾')
  })
})
