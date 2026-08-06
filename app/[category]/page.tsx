import { notFound } from 'next/navigation'
import QuestionView from '@/components/QuestionView'
import { listArticleTopics } from '@/lib/knowledge'
import { listCategories, loadCategory } from '@/lib/questions'

export const dynamic = 'force-dynamic'

export default async function CategoryPage({ params }: { params: Promise<{ category: string }> }) {
  const { category: raw } = await params
  const category = decodeURIComponent(raw)
  if (!listCategories().includes(category)) notFound()
  return (
    <QuestionView
      category={category}
      initialQuestions={loadCategory(category)}
      kbTopics={listArticleTopics(category)}
    />
  )
}
