import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // 生产(常驻服务)与开发用不同的构建目录,避免 npm run dev 冲掉常驻服务的产物。
  // 常驻服务通过 NEXT_DIST_DIR=.next-prod 启动,开发保持默认 .next。
  distDir: process.env.NEXT_DIST_DIR || '.next',
}

export default nextConfig
