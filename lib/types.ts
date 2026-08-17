export const SECTION_NAMES = ['题目', '要点', '答案', '知识点', '追问', 'Note'] as const
export type SectionName = (typeof SECTION_NAMES)[number]

export interface QuestionMeta {
  difficulty?: string
  tags: string[]
  company?: string
  topic?: string
  summary?: string
  mastered: boolean
  /** 高频题标记,页面上可切换,写回 frontmatter */
  highfreq: boolean
}

export interface Question {
  category: string
  file: string
  meta: QuestionMeta
  sections: Partial<Record<SectionName, string>>
  error?: string
}

export interface CategoryStat {
  name: string
  total: number
  mastered: number
}

export interface Stats {
  total: number
  mastered: number
  categories: CategoryStat[]
}
