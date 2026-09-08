import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // 开发、验证构建与 blue/green 生产槽互相隔离。具体目录只由受控脚本传入。
  distDir: process.env.NEXT_DIST_DIR || '.next',
}

export default nextConfig
