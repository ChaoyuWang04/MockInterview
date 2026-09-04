import Link from 'next/link'
import InterviewSession from '@/components/interview/InterviewSession'
import type { ResumeOption } from '@/components/interview/InterviewSession'
import { buildCorpus } from '@/lib/interview/corpus'
import { listResumes, loadProfile } from '@/lib/interview/resume'
import { listSessions } from '@/lib/interview/session'

export const dynamic = 'force-dynamic'

export default function InterviewPage() {
  const resumes = listResumes()
  const corpus = buildCorpus()
  const sessionCount = listSessions().length
  const questionPool = corpus.candidates.filter((c) => c.kind === 'question').length

  const options: ResumeOption[] = resumes.map((r) => ({
    slug: r.slug,
    name: r.name,
    role: r.role,
    chapters: r.chapters,
    isDefault: r.isDefault,
    hasProfile: loadProfile(r.slug) !== null,
  }))

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8 flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">模拟面试</h1>
        <div className="flex shrink-0 items-baseline gap-4 text-xs">
          <Link href="/interview/sessions" className="text-gray-500 hover:text-gray-900">
            📋 复盘记录{sessionCount > 0 ? ` · ${sessionCount} 场` : ''} →
          </Link>
          <Link href="/" className="text-gray-400 hover:text-gray-700">
            ← 首页
          </Link>
        </div>
      </div>

      {options.length === 0 ? (
        <p className="text-sm text-gray-500">
          还没有简历。在 <code className="bg-gray-100 px-1">interview/resumes/</code> 下放一份
          markdown,frontmatter 写上 <code className="bg-gray-100 px-1">章节:</code>(抽题的硬门禁)。
          写法见 docs/11-模拟面试系统.md。
        </p>
      ) : (
        <>
          <p className="mb-8 font-mono text-xs text-gray-400">
            题库 {questionPool} 道可抽 · 候选池 {corpus.stats.候选池大小} 个可问点 · 本地模型判卷
          </p>
          <InterviewSession resumes={options} />
        </>
      )}
    </main>
  )
}
