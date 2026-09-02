import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from 'vitest/config'

/**
 * 把 .env.local 读进 process.env。
 * Next.js 自己会加载这个文件,但 vitest 不会 —— 不做这一步,
 * `npm run interview:live` 会静默打到默认后端(localhost:1234)上,
 * 日志里看着一切正常,只是测的根本不是你以为的那个模型。
 * 已存在的环境变量优先,方便命令行临时覆盖。
 */
function loadEnvLocal() {
  const file = path.join(process.cwd(), '.env.local')
  if (!fs.existsSync(file)) return
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/)
    if (!m) continue
    const key = m[1]
    if (process.env[key] === undefined) {
      process.env[key] = m[2].replace(/^["']|["']$/g, '')
    }
  }
}
loadEnvLocal()

export default defineConfig({
  test: { environment: 'node', include: ['tests/**/*.test.ts'] },
})
