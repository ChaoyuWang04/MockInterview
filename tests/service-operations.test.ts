import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const projectRoot = path.resolve(import.meta.dirname, '..')
const serverScript = path.join(projectRoot, 'scripts/leet-server.sh')

function packageScripts(): Record<string, string> {
  return JSON.parse(fs.readFileSync(path.join(projectRoot, 'package.json'), 'utf8')).scripts
}

function runHelper(body: string, projectDir?: string): string {
  const root = projectDir ?? fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
  return execFileSync(
    'bash',
    ['-c', `source "$1"; LEETPREP_DIR="$2"; _leet_paths; ${body}`, 'test', serverScript, root],
    { encoding: 'utf8' },
  ).trim()
}

describe('常驻服务发布契约', () => {
  it('普通构建与生产发布槽完全分离', () => {
    const scripts = packageScripts()

    expect(scripts.build).toContain('.next-check')
    expect(scripts.build).not.toContain('.next-prod')
    expect(scripts['build:release']).toBeTruthy()
    expect(scripts.start).toContain('scripts/leet-server.sh start')
  })

  it('候选槽依据实际运行槽选择', () => {
    expect(runHelper('_leet_inactive_dist .next-blue')).toBe('.next-green')
    expect(runHelper('_leet_inactive_dist .next-green')).toBe('.next-blue')
    expect(runHelper('_leet_inactive_dist .next-prod')).toBe('.next-blue')
  })

  it('状态只接受固定白名单槽且要求完整构建', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
    fs.mkdirSync(path.join(root, '.next-blue'))
    fs.mkdirSync(path.join(root, '.leet'))
    fs.writeFileSync(path.join(root, '.next-blue/BUILD_ID'), 'blue-id\n')
    fs.writeFileSync(path.join(root, '.leet/active-dist'), '.next-blue\n')

    expect(runHelper('_leet_read_dist "$LEET_ACTIVE_DIST_FILE"', root)).toBe('.next-blue')

    fs.writeFileSync(path.join(root, '.leet/active-dist'), '../../tmp\n')
    expect(() => runHelper('_leet_read_dist "$LEET_ACTIVE_DIST_FILE"', root)).toThrow()

    fs.writeFileSync(path.join(root, '.leet/active-dist'), '.next-green\n')
    expect(() => runHelper('_leet_read_dist "$LEET_ACTIVE_DIST_FILE"', root)).toThrow()
  })

  it('server.dist 存在但损坏时禁止猜成 legacy', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
    for (const dist of ['.next-blue', '.next-prod']) {
      fs.mkdirSync(path.join(root, dist))
      fs.writeFileSync(path.join(root, dist, 'BUILD_ID'), `${dist}-id\n`)
    }
    fs.mkdirSync(path.join(root, '.leet'))
    fs.writeFileSync(path.join(root, '.leet/server.dist'), '../../tmp\n')

    expect(() => runHelper('_leet_pid_owned() { return 0; }; _leet_actual_dist 123', root)).toThrow()
  })

  it('active-dist 损坏时不构建也不启动', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
    fs.mkdirSync(path.join(root, '.leet'))
    fs.writeFileSync(path.join(root, '.leet/active-dist'), '.next-green\n')
    const output = runHelper(
      `_leet_owned_pid() { return 1; }
       _leet_build_dist() { echo build; }
       _leet_start_dist() { echo start; }
       _leet_start_impl >/dev/null 2>&1
       echo "rc:$?"`,
      root,
    )

    expect(output).toBe('rc:1')
  })

  it('活锁存在时拒绝第二个服务操作', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
    const output = runHelper(
      'ln -s $$ "$LEET_LOCK_PATH"; _leet_acquire_lock >/dev/null 2>&1; echo "rc:$?"',
      root,
    )

    expect(output).toBe('rc:1')
  })

  it('不完整锁不会被竞争者当场回收', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
    const output = runHelper(
      'mkdir "$LEET_LOCK_PATH"; _leet_acquire_lock >/dev/null 2>&1; echo "rc:$?"',
      root,
    )

    expect(output).toBe('rc:1')
  })

  it('候选尚未监听时仍可按进程身份和目标槽清理', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
    const trace = path.join(root, 'trace')
    const output = runHelper(
      `_leet_raw_pid() { echo 123; }
       _leet_pid_identity() { return 0; }
       _leet_read_dist() { echo .next-green; }
       _leet_terminate_pid() { echo "kill:$1" >>"${trace}"; }
       _leet_stop_candidate .next-green
       cat "${trace}"`,
      root,
    )

    expect(output).toBe('kill:123')
  })

  it('脚本包含互斥、原子状态、PID 归属和 CSS/JS 健康检查', () => {
    const script = fs.readFileSync(serverScript, 'utf8')

    expect(script).toContain('operation.lock')
    expect(script).toMatch(/ln -s .*LEET_LOCK/)
    expect(script).toContain('_leet_atomic_write')
    expect(script).toContain('_leet_pid_owned')
    expect(script).toContain('/_next/static/')
    expect(script).toMatch(/\.css/)
    expect(script).toMatch(/\.js/)
  })

  it('候选启动失败时按原实际槽回滚', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
    const trace = path.join(root, 'trace')
    const output = runHelper(
      `_leet_owned_pid() { echo 123; }
       _leet_actual_dist() { echo .next-blue; }
       _leet_read_dist() { echo .next-blue; }
       _leet_build_dist() { echo "build:$1" >>"${trace}"; }
       _leet_stop_impl() { echo stop >>"${trace}"; }
       _leet_stop_candidate() { echo stop >>"${trace}"; }
       _leet_start_dist() { echo "start:$1" >>"${trace}"; [ "$1" = .next-blue ]; }
       _leet_commit_active() { echo "active:$1" >>"${trace}"; }
       _leet_rebuild_impl >/dev/null 2>&1 || true
       cat "${trace}"`,
      root,
    )

    expect(output.split('\n')).toEqual([
      'build:.next-green',
      'stop',
      'start:.next-green',
      'stop',
      'start:.next-blue',
      'active:.next-blue',
    ])
  })

  it('候选构建失败发生在停止在线服务之前', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'leet-release-test-'))
    const trace = path.join(root, 'trace')
    const output = runHelper(
      `_leet_owned_pid() { echo 123; }
       _leet_actual_dist() { echo .next-blue; }
       _leet_read_dist() { echo .next-blue; }
       _leet_build_dist() { echo "build:$1" >>"${trace}"; return 1; }
       _leet_stop_impl() { echo stop >>"${trace}"; }
       _leet_rebuild_impl >/dev/null 2>&1 || true
       cat "${trace}"`,
      root,
    )

    expect(output).toBe('build:.next-green')
  })

  it('可由 zsh source 而不误执行命令分发', () => {
    const output = execFileSync(
      'zsh',
      ['-c', 'source "$1"; _leet_inactive_dist .next-blue; whence -w leet-start', 'test', serverScript],
      { encoding: 'utf8' },
    ).trim()

    expect(output.split('\n')).toEqual(['.next-green', 'leet-start: function'])
  })

  it('daemon 使用调用方指定的固定槽并记录实际槽', () => {
    const daemon = fs.readFileSync(path.join(projectRoot, 'scripts/leet-daemon.py'), 'utf8')

    expect(daemon).toContain('dist_dir')
    expect(daemon).toContain('server_dist_file')
    expect(daemon).toContain('NEXT_DIST_DIR=dist_dir')
    expect(daemon).not.toContain('NEXT_DIST_DIR=".next-prod"')
  })
})
