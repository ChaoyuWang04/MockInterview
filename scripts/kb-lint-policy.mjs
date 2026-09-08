import path from 'node:path'

const LONG_FORM_MARKER = /^> \*\*篇型\*\*:(?:总述篇|对比篇)$/

function hasCanonicalLongFormHeader(text) {
  const lines = text.split(/\r?\n/)
  return lines[0]?.startsWith('# ')
    && lines[1] === ''
    && lines[2]?.startsWith('一句话:')
    && lines[3] === ''
    && LONG_FORM_MARKER.test(lines[4] ?? '')
}

export function hasUnlimitedLength(file, text) {
  return path.basename(file) === '00-总览.md' || hasCanonicalLongFormHeader(text)
}

export function lengthWarning(file, text, lineCount) {
  if (lineCount < 100) return `篇幅 ${lineCount} 行,契约建议至少 120 行`
  if (lineCount > 300 && !hasUnlimitedLength(file, text)) {
    return `篇幅 ${lineCount} 行,契约建议 120–260`
  }
  return null
}
