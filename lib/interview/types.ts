/** 文章状态:能不能拿来出题,取决于它有没有正文 */
export type ArticleState = 'ready' | 'legacy' | 'placeholder'

/** 知识库「面试考点串联」表里的一行 */
export interface ExamPoint {
  /** 高频问法 */
  ask: string
  /** 本文哪一节 —— 答不上来时去看哪 */
  where: string
}

export interface ArticleEntry {
  /** 文章名,全局唯一,同时是题目 topic 第一段的匹配键 */
  title: string
  /** 章节,如 `04-Infra`(带 NN- 前缀,与简历 frontmatter 的 `章节:` 对齐) */
  chapter: string
  /** 相对 knowledge/ 的路径段,如 ['04-Infra','01-原理','KVCache.md'] */
  segments: string[]
  state: ArticleState
  /** 正文含 🔴 重点考点标记 */
  keypoint: boolean
  examPoints: ExamPoint[]
  /** 能不能当出题源:有正文(非占位)且有考点表 */
  usableAsSource: boolean
}

export interface QuestionEntry {
  /** `分类/文件名`,如 `AI Infra/001-gpu内存模型.md` */
  id: string
  category: string
  file: string
  /** frontmatter 的完整 topic */
  topic: string
  /** topic 第一段 = 对应的知识库文章名 */
  article: string
  summary: string
  tags: string[]
  difficulty?: string
  highfreq: boolean
  mastered: boolean
  /** tags 含「待校对」——答案是 AI 代写、尚未人工核对 */
  needsReview: boolean
  /** `## 要点` 的条目数,判卷用 */
  pointCount: number
  /** `## 追问` 的条目,现成的追问池(不用现场生成) */
  followUps: string[]
  /** 匹配到的文章所在章节;匹配不到则为 undefined,过不了章节门禁 */
  chapter?: string
  articleState?: ArticleState
}

/** 抽题候选:题库题与知识库考点行合并成同一个池 */
export interface Candidate {
  /** 题库题用 `q:<id>`,知识点用 `k:<文章名>#<行号>` */
  id: string
  kind: 'question' | 'exam-point'
  chapter: string
  /** 所属文章名,用于查亲和度与同 topic 衰减 */
  article: string
  /** 展示用的问法(题库题是 summary,考点是 ask) */
  ask: string
  highfreq: boolean
  mastered: boolean
}

export interface Corpus {
  questions: QuestionEntry[]
  articles: ArticleEntry[]
  candidates: Candidate[]
  stats: CorpusStats
}

export interface CorpusStats {
  题目总数: number
  参与出题的题目: number
  排除的分类: Record<string, number>
  匹配不到文章的题目: number
  文章总数: number
  可出题文章: number
  考点行总数: number
  候选池大小: number
  按章节: Record<string, number>
}
