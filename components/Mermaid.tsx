'use client'

import { useEffect, useId, useState } from 'react'

export default function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/[^a-zA-Z0-9]/g, '')
  const [svg, setSvg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const mermaid = (await import('mermaid')).default
        mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'strict' })
        const { svg } = await mermaid.render(`mmd${id}`, chart)
        if (!cancelled) setSvg(svg)
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [chart, id])

  if (error)
    return (
      <div className="mermaid-container border border-red-200 bg-red-50 p-3 text-sm">
        <pre className="overflow-x-auto whitespace-pre-wrap">{chart}</pre>
        <p className="mt-2 text-red-600">Mermaid 渲染失败:{error}</p>
      </div>
    )
  if (!svg) return <div className="mermaid-container text-sm text-gray-400">图表渲染中…</div>
  return <div className="mermaid-container" dangerouslySetInnerHTML={{ __html: svg }} />
}
