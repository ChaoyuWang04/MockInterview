'use client'

import { useCallback, useRef, useState } from 'react'
import { collectHotwords } from '@/components/interview/useVoice'

interface SttOption {
  id: string
  label: string
  ready: boolean
}

interface Result {
  model: string
  label: string
  text: string
  elapsedMs: number
  error?: string
}

/** 试录样句:中文骨架 + 密集英文术语。念这句最能暴露 STT 在中英混杂上的短板。 */
const SCRIPTS = [
  'moe 的 dispatch 有哪些方案?allgather、allreduce、all2all、deepep 通信量分别是多少?',
  '在 CUDA Graph 下能获取 KV cache 的量化参数吗?怎么获取?',
  'PD 分离、chunked prefill、ngram 投机解码,这几个在 vLLM 里怎么配合?',
]

/** 系统真实能拿到的热词:题目所属文章名 + 题干里的术语。只给规范写法,不展开变体。 */
const HOTWORDS = collectHotwords([
  'MoE', 'DeepEP', 'all2all', 'AllGather', 'AllReduce', 'dispatch', 'combine',
  'CUDA Graph', 'KV cache', 'vLLM', 'chunked prefill', 'ngram', 'PD 分离',
  'prefill', 'decode', 'kernel', 'shared memory', 'tiling', 'FlashAttention',
]).join(',')

/**
 * STT 对照台:用**你自己的声音**录一段,同一份音频喂给所有后端,
 * 结果并排看。这是决定 STT 选型的唯一可靠依据 ——
 * 拿 TTS 合成音测出来的排名不代表真人口音下的表现。
 */
export default function SttBench({ options }: { options: SttOption[] }) {
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState<Result[]>([])
  const [script, setScript] = useState(SCRIPTS[0])
  const [useHot, setUseHot] = useState(true)
  const [error, setError] = useState('')
  const [audioUrl, setAudioUrl] = useState('')

  const chunks = useRef<Blob[]>([])
  const recorder = useRef<MediaRecorder | null>(null)

  const runAll = useCallback(
    async (blob: Blob) => {
      setBusy(true)
      setResults([])
      const ready = options.filter((o) => o.ready)
      const out: Result[] = []
      for (const opt of ready) {
        const fd = new FormData()
        fd.append('file', blob, 'rec.webm')
        fd.append('model', opt.id)
        if (useHot) fd.append('hotwords', HOTWORDS)
        const t = Date.now()
        try {
          const res = await fetch('/api/interview/transcribe', { method: 'POST', body: fd })
          const data = await res.json()
          out.push({
            model: opt.id,
            label: opt.label,
            text: data.text ?? '',
            elapsedMs: data.elapsed_ms ?? Date.now() - t,
            error: data.error,
          })
        } catch (e) {
          out.push({ model: opt.id, label: opt.label, text: '', elapsedMs: 0, error: (e as Error).message })
        }
        setResults([...out])
      }
      setBusy(false)
    },
    [options, useHot],
  )

  const start = useCallback(async () => {
    setError('')
    setResults([])
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      chunks.current = []
      const mr = new MediaRecorder(stream)
      mr.ondataavailable = (e) => e.data.size > 0 && chunks.current.push(e.data)
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunks.current, { type: mr.mimeType })
        setAudioUrl(URL.createObjectURL(blob))
        void runAll(blob)
      }
      recorder.current = mr
      mr.start()
      setRecording(true)
    } catch (e) {
      setError(`拿不到麦克风:${(e as Error).message}`)
    }
  }, [runAll])

  const stop = useCallback(() => {
    recorder.current?.stop()
    setRecording(false)
  }, [])

  /**
   * 判断念稿里的英文术语有没有被转写出来。
   *
   * 归一化要做等价折叠,否则会误判:`all2all` / `all to all` / `All-to-All` / `alltoall`
   * 是同一个词的四种写法,只做「去空格后子串匹配」会把后三种全判成漏掉
   * (踩过 —— 一度让 Qwen3-ASR 被低估了一分)。
   */
  const norm = (x: string) =>
    x
      .toLowerCase()
      .replace(/[\s\-_.,、,。??!!]/g, '')
      .replace(/2/g, 'to')
      .replace(/4/g, 'for')
  const terms = script.match(/[A-Za-z][A-Za-z0-9]*\d*[A-Za-z0-9]*/g) ?? []
  const scoreOf = (text: string) => {
    const low = norm(text)
    const hit = terms.filter((t) => low.includes(norm(t)))
    return { hit: hit.length, total: terms.length, missed: terms.filter((t) => !hit.includes(t)) }
  }

  return (
    <section className="border border-gray-200 bg-white p-4">
      <h3 className="mb-1 text-sm font-semibold text-gray-700">STT 对照台</h3>
      <p className="mb-3 text-xs leading-relaxed text-gray-500">
        用你自己的声音念下面这句,同一段录音会喂给所有后端,结果并排对比。
        合成音测出来的排名不代表真人口音下的表现,这里才算数。
      </p>

      <label className="mb-3 block">
        <span className="mb-1 block text-xs text-gray-500">念这句</span>
        <select
          value={script}
          onChange={(e) => setScript(e.target.value)}
          className="w-full border border-gray-300 px-2 py-1.5 text-sm focus:border-gray-600 focus:outline-none"
        >
          {SCRIPTS.map((s) => (
            <option key={s} value={s}>
              {s.slice(0, 40)}…
            </option>
          ))}
        </select>
      </label>

      <div className="mb-3 border-l-2 border-gray-300 bg-gray-50 py-2 pl-3 text-sm leading-relaxed">
        {script}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={recording ? stop : start}
          disabled={busy}
          className={`border px-4 py-2 text-sm transition-colors disabled:opacity-40 ${
            recording
              ? 'border-red-600 bg-red-600 text-white hover:bg-red-500'
              : 'border-gray-800 bg-gray-900 text-white hover:bg-gray-700'
          }`}
        >
          {recording ? '⏹ 停止并转写' : busy ? '转写中…' : '● 开始录音'}
        </button>
        <label className="flex items-center gap-1.5 text-xs text-gray-600">
          <input type="checkbox" checked={useHot} onChange={(e) => setUseHot(e.target.checked)} />
          带热词
        </label>
        {audioUrl && <audio src={audioUrl} controls className="h-8" />}
      </div>

      {error && <p className="mt-3 bg-red-50 p-2 text-xs text-red-700">{error}</p>}

      {results.length > 0 && (
        <div className="mt-4 space-y-3">
          {results.map((r) => {
            const s = scoreOf(r.text)
            return (
              <div key={r.model} className="border border-gray-200 p-3">
                <div className="mb-1 flex items-baseline justify-between">
                  <span className="text-sm font-semibold">{r.label}</span>
                  <span className="font-mono text-[11px] text-gray-400">
                    {r.elapsedMs}ms · 术语 {s.hit}/{s.total}
                  </span>
                </div>
                {r.error ? (
                  <p className="text-xs text-red-600">{r.error}</p>
                ) : (
                  <>
                    <p className="text-sm leading-relaxed">{r.text || '(空)'}</p>
                    {s.missed.length > 0 && (
                      <p className="mt-1 font-mono text-[11px] text-amber-700">
                        漏掉:{s.missed.join(' · ')}
                      </p>
                    )}
                  </>
                )}
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}
