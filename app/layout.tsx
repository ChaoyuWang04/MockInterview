import type { Metadata } from 'next'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github.css'
import './globals.css'

export const metadata: Metadata = {
  title: '刷题系统',
  description: '本地大模型面试刷题系统',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen text-gray-900 antialiased">{children}</body>
    </html>
  )
}
