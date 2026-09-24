import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import ProjectReader from '@/components/ProjectReader'
import { getOsPages, isValidOsProject } from '@/lib/opensource'

export const dynamic = 'force-dynamic'

export async function generateMetadata({
  params,
}: {
  params: Promise<{ topic: string; project: string }>
}): Promise<Metadata> {
  const { project: rawP } = await params
  return { title: decodeURIComponent(rawP) }
}

export default async function OsProjectPage({
  params,
}: {
  params: Promise<{ topic: string; project: string }>
}) {
  const { topic: rawT, project: rawP } = await params
  const topic = decodeURIComponent(rawT)
  const project = decodeURIComponent(rawP)
  if (!isValidOsProject(topic, project)) notFound()
  return <ProjectReader topic={topic} project={project} pages={getOsPages(topic, project)} />
}
