import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { originalPdfHref, resolveOriginalPdf } from '../lib/original-pdf'

describe('原文 PDF', () => {
  it('已发布且磁盘有 PDF 的报告给出 confined 路径', () => {
    const file = resolveOriginalPdf({ lib: 'reports', company: 'DeepSeek', slug: 'DeepSeek-V3' })
    expect(file).toBe(path.join(process.cwd(), 'papers', 'DeepSeek', 'DeepSeek-V3.pdf'))
    expect(fs.existsSync(file!)).toBe(true)
    expect(originalPdfHref({ lib: 'reports', company: 'DeepSeek', slug: 'DeepSeek-V3' })).toBe(
      '/api/original-pdf?lib=reports&slug=DeepSeek-V3&company=DeepSeek',
    )
  })

  it('拒绝路径穿越和未发布 slug', () => {
    expect(resolveOriginalPdf({ lib: 'reports', company: '..', slug: 'DeepSeek-V3' })).toBeNull()
    expect(resolveOriginalPdf({ lib: 'reports', company: 'DeepSeek', slug: '..' })).toBeNull()
    expect(
      resolveOriginalPdf({ lib: 'reports', company: 'DeepSeek', slug: '../DeepSeek-V3' }),
    ).toBeNull()
    expect(resolveOriginalPdf({ lib: 'reports', company: 'DeepSeek', slug: '没有这篇' })).toBeNull()
  })

  it('没有本地 PDF 的已发布报告不给链接', () => {
    expect(resolveOriginalPdf({ lib: 'reports', company: 'DeepSeek', slug: 'DeepSeek-OCR-2' })).toBeNull()
    expect(originalPdfHref({ lib: 'reports', company: 'DeepSeek', slug: 'DeepSeek-OCR-2' })).toBeNull()
  })
})
