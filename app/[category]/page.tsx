import { notFound } from 'next/navigation'
import QuestionView from '@/components/QuestionView'
import { kbLinksFor } from '@/lib/knowledge'
import { listCategories, loadCategory } from '@/lib/questions'

export const dynamic = 'force-dynamic'

export default async function CategoryPage({ params }: { params: Promise<{ category: string }> }) {
  const { category: raw } = await params
  const category = decodeURIComponent(raw)
  if (!listCategories().includes(category)) notFound()
  const questions = loadCategory(category)
  // 知识库分章已不与题库分类一一对应,按 topic 第一段全库找文章
  const topics = questions.map((q) => q.meta.topic?.split('/')[0]?.trim() ?? '')
  return (
    <QuestionView category={category} initialQuestions={questions} kbLinks={kbLinksFor(topics)} />
  )
}
