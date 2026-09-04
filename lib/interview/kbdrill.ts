/**
 * 单篇过题:读完一篇知识库文章之后,把这一篇的题库题与考点行整个过一遍,
 * 用「说得出来」验证「真的看懂了」。
 *
 * 复用的是过题档的全套机器 —— 判卷、结算 UI、语音、会话落盘、复盘,一个字没改。
 * 这里只解决两件这个场景独有的事:
 *   ① 池子只限这一篇(不看简历章节门禁,也不看亲和度 —— 是你主动点进来的)
 *   ② 考点行没有人写的 `## 要点`,参考答案要从**文章原文**里按小节定位出来
 */
import { findArticle, getArticleBySegments } from '../knowledge'
import { loadArticles } from './corpus'
import type { PhaseMaterial } from './phases'

/** 考点候选的 id 形如 `k:<文章名>#<行号>`,和 corpus.ts 里生成的一致 */
export function parseKbId(id: string): { article: string; index: number } | null {
  const m = id.match(/^k:(.+)#(\d+)$/)
  return m ? { article: m[1], index: Number(m[2]) } : null
}

/** 中文数字 → 阿拉伯数字。文章小节是 `## 一、`/`## 二、` 这种写法 */
const CN_NUM: Record<string, number> = {
  一: 1, 二: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9, 十: 10,
}

const CN_DIGITS = Object.keys(CN_NUM)

/** 正文里所有 `## ` 小节的 [行号, 标题] */
function headings(lines: string[]): [number, string][] {
  const out: [number, string][] = []
  lines.forEach((l, i) => {
    if (/^##\s/.test(l)) out.push([i, l.replace(/^##\s*/, '').trim()])
  })
  return out
}

/**
 * 从「本文哪一节」那一格里解析出小节号。
 *
 * 契约确立后写的文章都是 `二(压缩过的答案)`,但**旧稿有三种别的写法**,
 * 各章不统一(实测:`第一节` / `2.3` / 长句末尾带 `(§二)` / 干脆只写小节标题的关键词)。
 * 一层层试,全都试不出来才回落关键词匹配。
 */
function sectionIndex(where: string, heads: [number, string][]): number {
  const w = where.trim()
  // ① `§二` / `(§二)` —— 旧稿把标记放在长句末尾
  const para = w.match(/§\s*([一二三四五六七八九十])/)
  if (para) return CN_NUM[para[1]]
  // ② `第二节`
  const nth = w.match(/第\s*([一二三四五六七八九十])\s*节/)
  if (nth) return CN_NUM[nth[1]]
  // ③ 开头就是中文数字(契约写法)
  if (CN_NUM[w[0]]) return CN_NUM[w[0]]
  // ④ 开头是阿拉伯数字,`2.3` 取整数位
  const ar = w.match(/^(\d+)/)
  if (ar) return Number(ar[1])
  // ⑤ 只写了小节标题的关键词 —— 拿第一个片段去标题里找
  const key = w.split(/[((;;,,、]/)[0].trim()
  if (key.length >= 2) {
    const i = heads.findIndex(([, title]) => title.includes(key))
    if (i >= 0) return i + 1
  }
  return 0
}

/**
 * 按考点表的「本文哪一节」定位原文小节。
 *
 * 那一格长这样:`二(capture 成 DAG → instantiate → replay;所有参数按值冻结)`——
 * 开头的中文数字就是小节号,括号里是压缩过的答案。
 *
 * **定位不到就返回空串**,由调用方回落到整篇正文。绝不猜、绝不编 —— 参考答案错了
 * 比没有更糟,它会直接把判分带偏。
 */
export function sectionOf(body: string, where: string): string {
  const lines = body.split('\n')
  const heads = headings(lines)
  const n = sectionIndex(where, heads)
  if (!n) return ''
  // 优先按 `## 一、` 这种带编号的标题找;找不到就按出现顺序取第 n 个 `## `
  let head = lines.findIndex((l) => new RegExp(`^##\\s*${CN_DIGITS[n - 1]}、`).test(l))
  if (head < 0) head = heads[n - 1]?.[0] ?? -1
  if (head < 0) return ''
  let end = lines.length
  for (let i = head + 1; i < lines.length; i++) {
    if (/^##\s/.test(lines[i])) {
      end = i
      break
    }
  }
  return lines.slice(head, end).join('\n').trim()
}

/** 读一篇文章的正文(不含 frontmatter 的原始 markdown) */
export function articleBody(title: string): string | null {
  const a = findArticle(title)
  if (!a) return null
  return getArticleBySegments(a.segments)
}

/**
 * 一条考点行的判分材料。
 *
 * **要点留空** —— 这一篇没有人写的清单,判卷时由模型从参考答案里现抽 3–5 条再对照,
 * 抽出来的清单会连同原文一起展示给你(过题档本来就当场给答案),清单靠不靠谱你自己一眼能判。
 * 这比「不打勾只给评价」多一个命中率数字,比「假装有人写过清单」诚实。
 */
export function examPointMaterial(article: string, index: number): PhaseMaterial | null {
  const entry = loadArticles().find((a) => a.title === article)
  const point = entry?.examPoints[index]
  if (!entry || !point) return null
  const body = articleBody(article) ?? ''
  const section = sectionOf(body, point.where)
  return {
    题目: point.ask,
    要点: [], // 空 = 没有人写的清单,判卷时现抽
    答案: section || body,
    追问: [],
    needsReview: false,
    selfPoints: true,
    article,
    chapter: entry.chapter,
    extra: [
      '### 这一题没有现成的要点清单',
      '',
      '上面的参考答案是**这篇文章的原文**(考点表指向的那一节)。',
      '判分前你先从原文里抽 **3–5 条**「答到什么才算说清楚了」,写进 `要点` 字段;',
      '再拿他的回答逐条对照,命中的序号进 `hit`,没命中的进 `miss`。',
      '',
      '- 要点只能来自上面的原文,**不要补原文没有的东西**',
      '- 每条一句话,是可判断的具体内容,不是「理解了原理」这种废话',
      `- 这条考点在原文里的位置:${point.where}`,
    ].join('\n'),
  }
}

/** 这篇文章能出多少问:题库题 + 考点行 */
export function articlePoolSize(title: string, questionCount: number): number {
  const entry = loadArticles().find((a) => a.title === title)
  return questionCount + (entry?.examPoints.length ?? 0)
}
