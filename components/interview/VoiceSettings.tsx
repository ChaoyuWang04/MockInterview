'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { VoiceConfig } from '@/components/interview/useVoice'

interface Catalog {
  up: boolean
  hint?: string
  tts: { id: string; label: string; kind: string; voices: string[]; ready: boolean }[]
  stt: { id: string; label: string; ready: boolean }[]
  defaults?: { tts: string; stt: string; voice: string; speed: number }
}

/** 试听样句:都是真题,术语密度从高到低 */
const SAMPLES = [
  'moe 的 dispatch 有哪些方案?allgather、allreduce、all2all、deepep 通信量分别是多少?',
  'PD 分离、chunked prefill、ngram 投机解码,这几个在 vLLM 里怎么配合?',
  '你简历里说 PrefixGrouper 端到端提速 2.31 倍,那个 87% 的重复率是怎么测出来的?',
]
const SAMPLE = SAMPLES[0]

/** 音色描述的起手式。都显式写了「标准普通话、无地方口音」—— 预设音色的口音问题就出在这。 */
const INSTRUCT_PRESETS = [
  { name: '成熟大叔', text: '四十五岁左右的男性,标准普通话,字正腔圆,没有任何地方口音,声音低沉浑厚,语速平缓从容,像一位资深技术专家' },
  { name: '成熟女士', text: '三十五到四十岁的女性,标准普通话,发音清晰标准,没有地方口音,声音沉稳知性,语速平稳,像一位技术主管' },
  { name: '冷静面试官', text: '三十多岁的男性,标准普通话,播音腔,吐字清楚,语气平静克制不带感情,像在严肃地问问题' },
  { name: '温和资深', text: '四十岁男性,标准普通话,声音温和但有分量,语速不快,像一位耐心的技术导师' },
]

export default function VoiceSettings({
  config,
  onChange,
}: {
  config: VoiceConfig
  onChange: (c: VoiceConfig) => void
}) {
  const [cat, setCat] = useState<Catalog | null>(null)
  const [busy, setBusy] = useState('')
  const [note, setNote] = useState('')
  const [custom, setCustom] = useState(SAMPLE)
  const [normalize, setNormalize] = useState(true)
  const [spoken, setSpoken] = useState('')
  const audioRef = useRef<AudioContext | null>(null)

  useEffect(() => {
    fetch('/api/interview/voice')
      .then((r) => r.json())
      .then(setCat)
      .catch(() => setCat({ up: false, tts: [], stt: [], hint: '取不到语音服务状态' }))
  }, [])

  const set = (patch: Partial<VoiceConfig>) => onChange({ ...config, ...patch })

  const preview = useCallback(
    async (text = SAMPLE) => {
      setBusy('preview')
      setNote('')
      try {
        const res = await fetch('/api/interview/voice', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input: text,
            model: config.ttsModel,
            voice: config.voice,
            instruct: config.instruct,
            speed: config.speed,
            temperature: config.temperature,
            normalize,
          }),
        })
        if (!res.ok) throw new Error(((await res.json()) as { error?: string }).error ?? `HTTP ${res.status}`)
        const ms = res.headers.get('X-Elapsed-Ms')
        const sec = res.headers.get('X-Audio-Seconds')
        const said = res.headers.get('X-Spoken')
        setSpoken(said ? decodeURIComponent(said) : '')
        // 用 Web Audio 而不是 <audio> —— 后者 volume 上限是 1,放大不了
        const buf = await res.arrayBuffer()
        const ctx = (audioRef.current ??= new AudioContext())
        if (ctx.state === 'suspended') await ctx.resume()
        const decoded = await ctx.decodeAudioData(buf)
        const src = ctx.createBufferSource()
        src.buffer = decoded
        const gain = ctx.createGain()
        gain.gain.value = config.volume
        src.connect(gain).connect(ctx.destination)
        src.start()
        setNote(`生成 ${ms}ms · 音频 ${sec}s`)
      } catch (e) {
        setNote((e as Error).message)
      } finally {
        setBusy('')
      }
    },
    [config, normalize],
  )

  const ttsSpec = cat?.tts.find((t) => t.id === config.ttsModel)
  const isDesign = ttsSpec?.kind === 'design'

  return (
    <section className="border border-gray-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={(e) => set({ enabled: e.target.checked })}
            disabled={!cat?.up}
          />
          语音模式
          <span className="font-normal text-xs text-gray-400">
            开:题目自动念出来,可以按住说话作答
          </span>
        </label>
        {cat && !cat.up && (
          <span className="font-mono text-[11px] text-amber-600">语音服务未启动</span>
        )}
      </div>

      {cat && !cat.up && (
        <p className="mb-3 bg-amber-50 p-2 font-mono text-[11px] leading-relaxed text-amber-800">
          {cat.hint ?? '语音服务不可达'}
        </p>
      )}

      {cat?.up && (
        <div className="space-y-3 text-sm">
          <div className="grid gap-3 sm:grid-cols-2">

            <label className="block">
              <span className="mb-1 block text-xs text-gray-500">
                语速 <span className="font-mono text-gray-700">{config.speed.toFixed(2)}×</span>
              </span>
              <input
                type="range"
                min={0.8}
                max={2}
                step={0.05}
                value={config.speed}
                onChange={(e) => set({ speed: Number(e.target.value) })}
                className="w-full"
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-gray-500">
                音量 <span className="font-mono text-gray-700">{Math.round(config.volume * 100)}%</span>
                {config.volume > 1 && <span className="ml-1 text-amber-600">增益</span>}
              </span>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={config.volume}
                onChange={(e) => set({ volume: Number(e.target.value) })}
                className="w-full"
              />
            </label>

            {isDesign && (
              <label className="block">
                <span className="mb-1 block text-xs text-gray-500">
                  音色稳定度{' '}
                  <span className="font-mono text-gray-700">T={config.temperature.toFixed(2)}</span>
                </span>
                <input
                  type="range"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={config.temperature}
                  onChange={(e) => set({ temperature: Number(e.target.value) })}
                  className="w-full"
                />
                <span className="font-mono text-[10px] text-gray-400">
                  越低两次生成越像,太低语调会发平
                </span>
              </label>
            )}

            <label className="block">
              <span className="mb-1 block text-xs text-gray-500">语音识别(STT)</span>
              <select
                value={config.sttModel}
                onChange={(e) => set({ sttModel: e.target.value })}
                className="w-full border border-gray-300 px-2 py-1.5 text-sm focus:border-gray-600 focus:outline-none"
              >
                {cat.stt.map((s) => (
                  <option key={s.id} value={s.id} disabled={!s.ready}>
                    {s.label}
                    {s.ready ? '' : '(不可用)'}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {isDesign && (
            <div>
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-xs text-gray-500">音色描述 · 用大白话写你想要的声音</span>
                <div className="flex flex-wrap gap-2">
                  {INSTRUCT_PRESETS.map((p) => (
                    <button
                      key={p.name}
                      onClick={() => set({ instruct: p.text })}
                      className="font-mono text-[11px] text-gray-400 hover:text-gray-700"
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>
              <textarea
                value={config.instruct}
                onChange={(e) => set({ instruct: e.target.value })}
                rows={2}
                placeholder="例:四十五岁男性,标准普通话,无地方口音,声音低沉,语速平缓"
                className="w-full border border-gray-300 p-2 text-sm leading-relaxed focus:border-gray-600 focus:outline-none"
              />
              <p className="mt-1 font-mono text-[11px] text-gray-400">
                预设音色带口音,所以描述里最好显式写「标准普通话、无地方口音」。
              </p>
            </div>
          )}

          <div className="border-t border-gray-100 pt-3">
            <div className="mb-1 flex items-baseline justify-between">
              <span className="text-xs text-gray-500">试听 · 想让它念什么就写什么</span>
              <div className="flex gap-2">
                {SAMPLES.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => setCustom(s)}
                    className="font-mono text-[11px] text-gray-400 hover:text-gray-700"
                  >
                    样句{i + 1}
                  </button>
                ))}
              </div>
            </div>
            <textarea
              value={custom}
              onChange={(e) => setCustom(e.target.value)}
              rows={3}
              className="w-full border border-gray-300 p-2 text-sm leading-relaxed focus:border-gray-600 focus:outline-none"
            />
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <button
                onClick={() => preview(custom)}
                disabled={busy === 'preview' || !custom.trim()}
                className="border border-gray-800 bg-gray-900 px-3 py-1.5 text-xs text-white transition-colors hover:bg-gray-700 disabled:opacity-40"
              >
                {busy === 'preview' ? '生成中…' : '▶ 念给我听'}
              </button>
              <label
                className="flex items-center gap-1.5 text-xs text-gray-600"
                title="不勾:按字面念,all2all 会念成「all 二 all」、deepep 当一个词念。勾上:先改写成 all to all / Deep E P 再念。只影响声音,判卷用的始终是原文。"
              >
                <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} />
                读音修正
                <span className="text-gray-400">(all2all → all to all)</span>
              </label>
              {note && <span className="font-mono text-[11px] text-gray-400">{note}</span>}
            </div>
            {spoken && spoken !== custom && (
              <p className="mt-2 bg-gray-50 p-2 font-mono text-[11px] leading-relaxed text-gray-600">
                实际念的是:{spoken}
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
