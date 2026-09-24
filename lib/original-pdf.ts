/**
 * 长文页「原文 PDF」:只给已发布解读对应的本地原件。
 * 报告 papers/<公司>/<slug>.pdf;研读 readings/_src/<方向>/<slug>.pdf。
 * 网页原件没有文件,返回 null。
 */
import fs from 'node:fs'
import path from 'node:path'

import { getReading } from './readings'
import { getReport } from './reports'

export type OriginalPdfQuery =
  | { lib: 'reports'; company: string; slug: string }
  | { lib: 'readings'; topic: string; slug: string }

function isSafeSegment(value: string): boolean {
  if (!value || value === '.' || value === '..') return false
  if (value.includes('\0') || value.includes('/') || value.includes('\\')) return false
  return true
}

function confined(root: string, ...parts: string[]): string | null {
  if (parts.some((part) => !isSafeSegment(part))) return null
  const resolvedRoot = path.resolve(root)
  const resolved = path.resolve(resolvedRoot, ...parts)
  const prefix = resolvedRoot.endsWith(path.sep) ? resolvedRoot : resolvedRoot + path.sep
  if (resolved !== resolvedRoot && !resolved.startsWith(prefix)) return null
  return resolved
}

export function resolveOriginalPdf(query: OriginalPdfQuery): string | null {
  if (query.lib === 'reports') {
    if (!getReport(query.company, query.slug)) return null
    const file = confined(path.join(process.cwd(), 'papers'), query.company, `${query.slug}.pdf`)
    if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) return null
    return file
  }
  if (!getReading(query.topic, query.slug)) return null
  const file = confined(
    path.join(process.cwd(), 'readings', '_src'),
    query.topic,
    `${query.slug}.pdf`,
  )
  if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) return null
  return file
}

export function originalPdfHref(query: OriginalPdfQuery): string | null {
  if (!resolveOriginalPdf(query)) return null
  const params = new URLSearchParams({ lib: query.lib, slug: query.slug })
  if (query.lib === 'reports') params.set('company', query.company)
  else params.set('topic', query.topic)
  return `/api/original-pdf?${params.toString()}`
}
