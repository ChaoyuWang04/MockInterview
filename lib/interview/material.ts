/**
 * 判分材料的统一入口:题库题和阶段题(自我介绍 / 项目深挖)走同一条路。
 *
 * 存在的理由是不让阶段机污染下游 —— `turn` 和 `session` 两条路由拿到的都是
 * 同一个形状,不必为「这是不是项目题」分叉。
 */
import { loadQuestionDetail } from './corpus'
import { examPointMaterial, parseKbId } from './kbdrill'
import { introMaterial, parsePhaseId, projectMaterial, type PhaseMaterial } from './phases'
import type { ProfileFile, ResumeEntry } from './resume'

export function resolveMaterial(
  id: string,
  resume: ResumeEntry,
  profile: ProfileFile | null,
  /** 会话种子:决定开场材料里项目清单的呈现顺序,整场不变 */
  seed = 0,
): PhaseMaterial | null {
  const phase = parsePhaseId(id)
  if (phase?.kind === 'intro') return introMaterial(profile, seed)
  if (phase?.kind === 'project') return projectMaterial(resume, profile, phase.index)

  // 单篇过题的考点行:参考答案从文章原文按小节捞,要点留空由模型现抽
  const kb = parseKbId(id)
  if (kb) return examPointMaterial(kb.article, kb.index)

  const d = loadQuestionDetail(id)
  if (!d) return null
  return {
    题目: d.题目,
    要点: d.要点,
    答案: d.答案,
    追问: d.追问,
    needsReview: d.entry.needsReview,
    article: d.entry.article,
    chapter: d.entry.chapter ?? '',
  }
}
