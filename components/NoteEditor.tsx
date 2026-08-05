'use client'

import { useState } from 'react'
import Markdown from './Markdown'

interface Props {
  note: string
  onSave: (note: string) => Promise<boolean>
}

export default function NoteEditor({ note, onSave }: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(note)
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    const ok = await onSave(draft)
    setSaving(false)
    if (ok) setEditing(false)
  }

  return (
    <section className="mt-8 border-t border-dashed border-gray-200 pt-6">
      <div className="flex items-center justify-between">
        <h3 className="font-mono text-xs tracking-widest text-gray-400">NOTE</h3>
        {!editing && (
          <button
            onClick={() => {
              setDraft(note)
              setEditing(true)
            }}
            className="text-sm text-blue-600 hover:underline"
          >
            编辑
          </button>
        )}
      </div>
      {editing ? (
        <div className="mt-3">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={6}
            autoFocus
            placeholder="支持 markdown 语法"
            className="w-full border border-gray-300 bg-white p-3 font-mono text-sm focus:border-blue-500 focus:outline-none"
          />
          <div className="mt-2 flex gap-3">
            <button
              onClick={save}
              disabled={saving}
              className="bg-gray-900 px-4 py-1.5 text-sm text-white disabled:opacity-50"
            >
              {saving ? '保存中…' : '保存'}
            </button>
            <button onClick={() => setEditing(false)} className="text-sm text-gray-500 hover:text-gray-900">
              取消
            </button>
          </div>
        </div>
      ) : note.trim() ? (
        <div className="prose prose-sm mt-3 max-w-none">
          <Markdown>{note}</Markdown>
        </div>
      ) : (
        <p className="mt-3 text-sm text-gray-300">…</p>
      )}
    </section>
  )
}
