'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import Mermaid from './Mermaid'

export default function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex, [rehypeHighlight, { plainText: ['mermaid'] }]]}
      components={{
        code({ node: _node, className, children: code, ...props }) {
          if (className?.includes('language-mermaid')) return <Mermaid chart={String(code ?? '')} />
          return (
            <code className={className} {...props}>
              {code}
            </code>
          )
        },
      }}
    >
      {children}
    </ReactMarkdown>
  )
}
