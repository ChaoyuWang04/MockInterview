import fs from 'node:fs'
import { NextResponse } from 'next/server'

import { resolveOriginalPdf, type OriginalPdfQuery } from '@/lib/original-pdf'

export const dynamic = 'force-dynamic'

function parseQuery(url: URL): OriginalPdfQuery | null {
  const lib = url.searchParams.get('lib')
  const slug = url.searchParams.get('slug')
  if (!slug) return null
  if (lib === 'reports') {
    const company = url.searchParams.get('company')
    if (!company) return null
    return { lib, company, slug }
  }
  if (lib === 'readings') {
    const topic = url.searchParams.get('topic')
    if (!topic) return null
    return { lib, topic, slug }
  }
  return null
}

export async function GET(req: Request) {
  const query = parseQuery(new URL(req.url))
  if (!query) return new NextResponse('Bad Request', { status: 400 })
  const file = resolveOriginalPdf(query)
  if (!file) return new NextResponse('Not Found', { status: 404 })

  const body = fs.readFileSync(file)
  return new NextResponse(body, {
    headers: {
      'Content-Type': 'application/pdf',
      'Content-Disposition': `inline; filename="${query.slug}.pdf"`,
      'Cache-Control': 'private, max-age=0, must-revalidate',
    },
  })
}
