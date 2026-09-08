import { describe, expect, it } from 'vitest'
import { lengthWarning } from '../scripts/kb-lint-policy.mjs'

describe('知识库篇幅策略', () => {
  const longForm = (type: '总述篇' | '对比篇') => `# 标题\n\n一句话:定位。\n\n> **篇型**:${type}`

  it('普通长文保留上限提醒', () => {
    expect(lengthWarning('knowledge/机制.md', '# 机制', 301)).toBe('篇幅 301 行,契约建议 120–260')
  })

  it('显式总述篇、对比篇和 00-总览 取消上限提醒', () => {
    expect(lengthWarning('knowledge/体系.md', longForm('总述篇'), 301)).toBeNull()
    expect(lengthWarning('knowledge/方案.md', longForm('对比篇'), 301)).toBeNull()
    expect(lengthWarning('knowledge/00-总览.md', '# 总览', 301)).toBeNull()
  })

  it('特殊篇型仍保留短文下限提醒', () => {
    expect(lengthWarning('knowledge/体系.md', longForm('总述篇'), 99)).toBe('篇幅 99 行,契约建议至少 120 行')
    expect(lengthWarning('knowledge/00-总览.md', '# 总览', 99)).toBe('篇幅 99 行,契约建议至少 120 行')
  })

  it('只靠文件名带“对比”不能冒充特殊篇型', () => {
    expect(lengthWarning('knowledge/未声明对比.md', '# 未声明对比', 301)).toBe('篇幅 301 行,契约建议 120–260')
  })

  it('正文末尾或代码块里的标记不能绕过上限', () => {
    const appended = '# 机制\n\n一句话:定位。\n\n正文\n\n> **篇型**:对比篇'
    const fenced = '# 机制\n\n一句话:定位。\n\n```markdown\n> **篇型**:对比篇\n```'
    expect(lengthWarning('knowledge/机制.md', appended, 301)).toBe('篇幅 301 行,契约建议 120–260')
    expect(lengthWarning('knowledge/机制.md', fenced, 301)).toBe('篇幅 301 行,契约建议 120–260')
  })
})
