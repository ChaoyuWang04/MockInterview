/**
 * 面试档的阶段机。
 *
 * 为什么要有它:改之前面试档只是「过题档不给答案」—— 一上来就问技术细节,
 * 没有自我介绍、没有项目背景、没有决策链。真面试不是那样走的,而且那样的话
 * 两个档位在做同一件事,面试档就没有存在意义了。
 *
 * **过题档不受影响**:它就该是纯技术细节训练,是另一个方向。
 *
 * 边界纪律和别处一致:**阶段由代码控制,阶段内追问归模型**。
 * 模型可以发 `next: nextphase` 提前结束一个阶段,但轮次上限是代码卡死的 ——
 * 否则一场面试可能在自我介绍上耗掉二十轮。
 */
import type { ProfileFile, ResumeEntry } from './resume'

export type InterviewPhase = 'intro' | 'project' | 'tech' | 'breadth'

export const PHASE_ORDER: InterviewPhase[] = ['intro', 'project', 'tech', 'breadth']

export const PHASE_LABEL: Record<InterviewPhase, string> = {
  intro: '开场',
  project: '项目深挖',
  tech: '技术延伸',
  breadth: '广度补充',
}

/**
 * 每个阶段最多几轮 —— 代码的硬上限,模型只能提前结束不能拖长。
 * `project` 是**每个项目**的上限,不是整个阶段的。
 */
export const PHASE_CAPS: Record<InterviewPhase, number> = {
  intro: 2,
  project: 5,
  tech: 8,
  breadth: Infinity,
}

/** 项目深挖的判分维度。**它就是这类题的 `## 要点`**,所以判卷内核一个字都不用改。 */
export const PROJECT_DIMENSIONS = [
  '背景:为什么做这件事,要解决的问题和目标是什么',
  '你的角色:这个项目里他具体负责哪一部分,不是团队做了什么',
  '关键决策:做了哪些技术选择(选了什么方案、什么模型、什么架构)',
  '权衡取舍:为什么不选别的路,选了这条的代价是什么',
  '量化结果:用数字说明效果,并说清这个数字怎么测出来的',
]

export const INTRO_ASK =
  '我们开始吧。先花两三分钟做个自我介绍,重点讲你最近在做的事和你在里面的角色。'

/** 一道题的判分材料,和 `corpus.ts` 的 `QuestionDetail` 同形 —— 好让下游代码不用分叉 */
export interface PhaseMaterial {
  题目: string
  要点: string[]
  答案: string
  追问: string[]
  needsReview: boolean
  /** 页面上的角标与热词都用它 */
  article: string
  chapter: string
  /** 追加进材料块的额外说明(开场用它给出项目清单) */
  extra?: string
  /**
   * 要点为空**但仍要判分** —— 由模型从参考答案原文现抽清单(单篇过题的考点行)。
   * 不加这个标记的话,materialBlock 会把它和「自我介绍不判分」混为一谈。
   */
  selfPoints?: boolean
}

/**
 * 从简历正文里抠出某个项目的原文段落(`### <项目名>` 到下一个同级标题)。
 *
 * 抠不出来就返回空串,调用方回落到整份简历 —— **绝不编造项目内容**,
 * 那正是这个系统三道防幻觉闸想挡的东西。
 */
export function projectSection(resumeBody: string, projectName: string): string {
  const lines = resumeBody.split('\n')
  const start = lines.findIndex(
    (l) => /^#{2,4}\s/.test(l) && l.replace(/^#+\s*/, '').trim() === projectName.trim(),
  )
  if (start < 0) return ''
  const level = (lines[start].match(/^#+/) ?? ['###'])[0].length
  let end = lines.length
  for (let i = start + 1; i < lines.length; i++) {
    const m = lines[i].match(/^(#+)\s/)
    if (m && m[1].length <= level) {
      end = i
      break
    }
  }
  return lines.slice(start, end).join('\n').trim()
}

/**
 * 会话种子决定的项目呈现顺序。
 *
 * **为什么必须由代码注入随机**:判卷调用是 `temperature: 0`,模型是确定性的 ——
 * 直接问它「你对哪个项目感兴趣」,同一份简历每次都会选同一个,多跑几场毫无意义。
 * 打乱呈现顺序之后,即使模型的偏好是「挑第一个」,每场也会落到不同项目上。
 *
 * 种子由前端在开场时生成、整场不变并随每次请求回传 —— 所以**同一场里顺序是稳定的**,
 * 材料块不会变,前缀缓存不受影响。
 */
export function projectOrder(count: number, seed: number): number[] {
  const idx = Array.from({ length: count }, (_, i) => i)
  // mulberry32,和 sampler.ts 的 seededRandom 同一套,保证可复现
  let s = seed >>> 0
  const rnd = () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  return idx
}

/**
 * 自我介绍。**不判分** —— 要点为空,hit/miss 自然都是空数组。
 *
 * 材料块里附上项目清单,让模型**听完自我介绍之后自己挑**一个想深挖的(`proj` 字段)。
 * 真面试官就是这么做的:顺着你刚讲的东西往下追,而不是按简历从上到下过一遍。
 * 你每次自我介绍讲的重点不同,它的选择自然就不同 —— 这是变化的主要来源。
 */
export function introMaterial(profile: ProfileFile | null, seed = 0): PhaseMaterial {
  const probes = profile?.项目深挖点 ?? []
  const order = projectOrder(probes.length, seed)
  const list = order.map((i) => `- \`proj: ${i}\` ${probes[i].项目}`)
  return {
    题目: INTRO_ASK,
    要点: [],
    答案: '',
    追问: [],
    needsReview: false,
    article: '开场',
    chapter: '面试流程',
    extra: probes.length
      ? [
          '### 他简历上的项目',
          '',
          '听完自我介绍,**挑一个你最想深挖的**,用 `proj` 填它的序号。',
          '挑的理由应该来自他刚才说的话 —— 哪个他讲得含糊、哪个数字可疑、哪个和岗位最相关。',
          '**不必按顺序,也不必每个都问。** 真面试官会挖透一两个,而不是每个都问一遍。',
          '',
          ...list,
        ].join('\n')
      : undefined,
  }
}

/**
 * 第 N 个项目的深挖材料。
 *
 * 判分依据是**他简历上自己写的东西 + 五个固定维度**,不是某份标准答案 ——
 * 这类题考的是「能不能把自己做过的事讲清楚」,不是「记不记得住知识点」。
 */
export function projectMaterial(
  resume: ResumeEntry,
  profile: ProfileFile | null,
  index: number,
): PhaseMaterial | null {
  const probe = profile?.项目深挖点?.[index]
  if (!probe) return null
  const section = projectSection(resume.body, probe.项目)
  return {
    题目: `聊聊「${probe.项目}」这个项目吧,先说说背景 —— 这件事是怎么来的,你在里面做什么?`,
    要点: PROJECT_DIMENSIONS,
    // 「参考答案」= 简历原文。模型据此判断他说的和写的对不对得上、有没有讲到位
    答案: section || '(简历里没有找到这个项目的段落,以上面的简历全文为准)',
    追问: [probe.可问],
    needsReview: false,
    article: probe.项目,
    chapter: '项目深挖',
    extra: [
      '### 这个阶段的自由裁量',
      '',
      '**追几轮、追哪个方向,完全由你定。** 五个维度是判分用的清单,不是提问的顺序 ——',
      '他哪一维讲得虚就往哪追,数字可疑就问怎么测的,决策含糊就问为什么不选别的路。',
      '挖透了发 `nextphase`;想换个项目就在 `proj` 里填另一个序号。',
      '',
      '**两条不许犯的**:',
      '',
      '1. **同一个问题不要问第二遍。** 他没答上来就换个角度或者往下走,',
      '   一字不差地再问一遍只会浪费两个人的时间',
      '2. **往上追,不要往下钻。** 追问该走向「为什么这么选、代价是什么、换个规模还成不成立」,',
      '   不是「那个参数具体设了多少、那个函数叫什么」。**问到实现细节就说明该换话题了** ——',
      '   发 `nextphase` 进技术延伸,那里才是问细节的地方',
    ].join('\n'),
  }
}

/** 当前项目关联到的知识库文章名 —— 技术延伸阶段就从这些文章的题里抽 */
export function relatedTopics(profile: ProfileFile | null, index: number): string[] {
  return profile?.项目深挖点?.[index]?.关联topic ?? []
}

/** 有几个项目可深挖 */
export function projectCount(profile: ProfileFile | null): number {
  return profile?.项目深挖点?.length ?? 0
}

/**
 * 合成 id。前端把它当普通题目 id 回传,服务端据此复原判分材料 ——
 * 和题库题走同一条路,`turn` 路由不用为阶段分叉。
 */
export const INTRO_ID = 'phase:intro'
export const projectId = (i: number) => `phase:project:${i}`

export function parsePhaseId(id: string): { kind: 'intro' } | { kind: 'project'; index: number } | null {
  if (id === INTRO_ID) return { kind: 'intro' }
  const m = id.match(/^phase:project:(\d+)$/)
  return m ? { kind: 'project', index: Number(m[1]) } : null
}
