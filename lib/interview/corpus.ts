import fs from 'node:fs'
import path from 'node:path'
import { isValidRef, listCategories, listQuestionFiles, loadQuestion, questionsRoot } from '../questions'
import { flattenArticles, knowledgeRoot, listKbTree } from '../knowledge'
import type {
  ArticleEntry,
  ArticleState,
  Candidate,
  Corpus,
  CorpusStats,
  ExamPoint,
  QuestionEntry,
} from './types'

/**
 * 整类排除的题库分类。
 * `手撕代码` 要当场写代码,不适合口头面试;具体数量由候选池统计现场输出。
 */
export const EXCLUDED_CATEGORIES = ['手撕代码']

/** 从正文抽「面试考点串联」表。成文篇用 markdown 表格,旧稿用编号列表,两种都认。 */
export function extractExamPoints(body: string): ExamPoint[] {
  const start = body.search(/^##\s+.*面试考点串联/m)
  if (start < 0) return []
  // 从该标题起,到下一个 ## 之前
  const section = body.slice(start).split(/^## /m)[1] ?? ''
  const out: ExamPoint[] = []
  for (const raw of section.split('\n')) {
    const line = raw.trim()
    if (line.startsWith('|')) {
      const cells = line.split('|').slice(1, -1).map((c) => c.trim())
      if (cells.length < 2) continue
      if (/^-+$/.test(cells[0].replace(/[:\s]/g, ''))) continue // 分隔行
      if (/^(高频)?问法$/.test(cells[0])) continue // 表头
      if (!cells[0]) continue
      out.push({ ask: cells[0], where: cells[1] })
      continue
    }
    // 旧稿:`1. 问法 →「小节」`
    const m = line.match(/^\d+\.\s*(.+?)\s*→\s*[「『"]?(.*?)[」』"]?\s*$/)
    if (m && m[1]) out.push({ ask: m[1], where: m[2] || '' })
  }
  return out
}

/** `- ` 开头的条目数(用于数 `## 要点` 有几条) */
function bulletCount(section: string | undefined): number {
  return section ? (section.match(/^-\s+/gm) ?? []).length : 0
}

function bulletItems(section: string | undefined): string[] {
  if (!section) return []
  return section
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.startsWith('- '))
    .map((l) => l.slice(2).trim())
    .filter(Boolean)
}

function stateOf(placeholder: boolean, legacy: boolean): ArticleState {
  if (placeholder) return 'placeholder'
  if (legacy) return 'legacy'
  return 'ready'
}

/**
 * 各章的 `00-总览.md` 是 hub 导航页,不是知识点专篇:
 * 剥掉 NN- 前缀后它们全部叫「总览」,标题会撞车;而且知识库地图明确写了
 * 「00-总览 不参与匹配」。所以整体排除,既不出题也不当匹配键。
 */
function isHubPage(segments: string[]): boolean {
  return segments[segments.length - 1] === '00-总览.md'
}

export function loadArticles(root = knowledgeRoot()): ArticleEntry[] {
  return flattenArticles(listKbTree(root))
    .filter((a) => !isHubPage(a.segments))
    .map((a) => {
      const body = fs.readFileSync(path.join(root, ...a.segments), 'utf8')
      const state = stateOf(a.placeholder, a.legacy)
      const examPoints = extractExamPoints(body)
      return {
        title: a.title,
        chapter: a.segments[0] ?? '',
        segments: a.segments,
        state,
        keypoint: a.keypoint,
        examPoints,
        // 占位稿没有正文,拿它出题就是让 AI 现编答案(铁律 4)
        usableAsSource: state !== 'placeholder' && examPoints.length > 0,
      }
    })
}

export function loadQuestions(root = questionsRoot()): QuestionEntry[] {
  const out: QuestionEntry[] = []
  for (const category of listCategories(root)) {
    for (const file of listQuestionFiles(category, root)) {
      const q = loadQuestion(category, file, root)
      if (q.error) continue
      const topic = q.meta.topic ?? ''
      out.push({
        id: `${category}/${file}`,
        category,
        file,
        topic,
        article: topic.split('/')[0]?.trim() ?? '',
        summary: q.meta.summary ?? '',
        tags: q.meta.tags,
        difficulty: q.meta.difficulty,
        highfreq: q.meta.highfreq,
        mastered: q.meta.mastered,
        needsReview: q.meta.tags.includes('待校对'),
        pointCount: bulletCount(q.sections['要点']),
        followUps: bulletItems(q.sections['追问']),
      })
    }
  }
  return out
}

export interface QuestionDetail {
  entry: QuestionEntry
  /** 题干原文 —— 主问念它,不改写(既保留真题口气,TTS 缓存又能 100% 命中) */
  题目: string
  /** 评分细则,逐条比对 */
  要点: string[]
  /** 判卷标准。**给模型看,但绝不念出来** */
  答案: string
  /** 手写好的追问池:优先查表,查不到才生成 */
  追问: string[]
}

/** 按 `分类/文件名` 读一道题的全文;L3 渐进式装载只在换题时调它一次 */
export function loadQuestionDetail(id: string, root = questionsRoot()): QuestionDetail | null {
  const slash = id.indexOf('/')
  if (slash < 0) return null
  const category = id.slice(0, slash)
  const file = id.slice(slash + 1)
  if (!isValidRef(category, file, root)) return null // 白名单校验,杜绝路径穿越
  const q = loadQuestion(category, file, root)
  if (q.error) return null
  const entry = loadQuestions(root).find((e) => e.id === id)
  if (!entry) return null
  return {
    entry,
    题目: (q.sections['题目'] ?? '').trim(),
    要点: bulletItems(q.sections['要点']),
    答案: (q.sections['答案'] ?? '').trim(),
    追问: bulletItems(q.sections['追问']),
  }
}

/**
 * 直接从题库取几条真题原话,当面试官口气的 few-shot。
 * 题库已经是唯一真源,不再维护另一套“原题档案”。
 */
export function loadToneSamples(limit = 6, root = questionsRoot()): string[] {
  const out: string[] = []
  for (const category of listCategories(root)) {
    for (const file of listQuestionFiles(category, root)) {
      const q = loadQuestion(category, file, root)
      const firstLine = q.sections['题目']?.split('\n').map((line) => line.trim()).find(Boolean)
      if (firstLine && firstLine.length > 8) out.push(firstLine)
      if (out.length >= limit) return out
    }
  }
  return out
}

/**
 * 合并成一个候选池。
 * 题库题自带 `## 要点`(评分细则)与 `## 追问`(现成追问池),所以是主力;
 * 知识库考点行只有问法与指向,用来补题库覆盖不到的章节。
 */
export function buildCorpus(
  qRoot = questionsRoot(),
  kRoot = knowledgeRoot(),
): Corpus {
  const articles = loadArticles(kRoot)
  const byTitle = new Map(articles.map((a) => [a.title, a]))
  const questions = loadQuestions(qRoot)

  const excluded: Record<string, number> = {}
  let unmatched = 0
  const candidates: Candidate[] = []

  for (const q of questions) {
    const article = byTitle.get(q.article)
    q.chapter = article?.chapter
    q.articleState = article?.state
    if (EXCLUDED_CATEGORIES.includes(q.category)) {
      excluded[q.category] = (excluded[q.category] ?? 0) + 1
      continue
    }
    if (!article) {
      unmatched++
      continue
    }
    candidates.push({
      id: `q:${q.id}`,
      kind: 'question',
      chapter: article.chapter,
      article: article.title,
      ask: q.summary,
      highfreq: q.highfreq,
      mastered: q.mastered,
    })
  }

  // 已经被题库题覆盖的文章,考点行只作判卷参考,不再重复出题
  const coveredArticles = new Set(candidates.map((c) => c.article))
  for (const a of articles) {
    if (!a.usableAsSource || coveredArticles.has(a.title)) continue
    a.examPoints.forEach((p, i) => {
      candidates.push({
        id: `k:${a.title}#${i}`,
        kind: 'exam-point',
        chapter: a.chapter,
        article: a.title,
        ask: p.ask,
        highfreq: false,
        mastered: false,
      })
    })
  }

  const byChapter: Record<string, number> = {}
  for (const c of candidates) byChapter[c.chapter] = (byChapter[c.chapter] ?? 0) + 1

  const stats: CorpusStats = {
    题目总数: questions.length,
    参与出题的题目: candidates.filter((c) => c.kind === 'question').length,
    排除的分类: excluded,
    匹配不到文章的题目: unmatched,
    文章总数: articles.length,
    可出题文章: articles.filter((a) => a.usableAsSource).length,
    考点行总数: articles.reduce((s, a) => s + a.examPoints.length, 0),
    候选池大小: candidates.length,
    按章节: byChapter,
  }

  return { questions, articles, candidates, stats }
}
