export const SECTION_NAMES = ['题目', '要点', '答案', '知识点', '追问', 'Note'] as const
export type SectionName = (typeof SECTION_NAMES)[number]

export interface QuestionMeta {
  difficulty?: string
  tags: string[]
  company?: string
  topic?: string
  summary?: string
  mastered: boolean
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
