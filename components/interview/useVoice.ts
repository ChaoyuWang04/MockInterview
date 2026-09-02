'use client'

import { useCallback, useRef } from 'react'

export interface VoiceConfig {
  ttsModel: string
  /** 预设音色名(CustomVoice 模型用) */
  voice: string
  /** 音色的文字描述(VoiceDesign 模型用)。预设音色都带口音,所以默认显式要求标准普通话 */
  instruct: string
  speed: number
  /** 播放增益。1 = 原样;可以超过 1(走 Web Audio 的 GainNode,<audio> 的 volume 上限是 1) */
  volume: number
  /** 采样温度。同一段音色描述,温度越高两次生成的差别越大。0.4 是实测折中点 */
  temperature: number
  sttModel: string
  enabled: boolean
}

export const DEFAULT_VOICE_CONFIG: VoiceConfig = {
  ttsModel: 'qwen3-tts-design',
  voice: '',
  instruct: '三十多岁的男性,标准普通话,播音腔,吐字清楚,语气平静克制不带感情,像在严肃地问问题',
  speed: 1.2,
  volume: 1.2,
  // 0.1:同一段描述两次生成尽量像。模型原默认 0.9 抖得厉害
  temperature: 0.1,
  // 实测(真声,3 句 15 术语,带热词):Whisper 14/15 · Qwen3-ASR 13/15 · Fun-ASR 10/15
  sttModel: 'whisper-turbo',
  enabled: true,
}

/**
 * 把一批术语展开成热词表。
 *
 * 同一个词给多种写法能明显提高召回:实测 `deepep` 只给 `DeepEP` 时,
 * Qwen3-ASR 听成 `DeepHP`;补上小写和拆写(`deepep` / `Deep EP`)之后两家都认了。
 * 驼峰词尤其需要 —— 模型听到的是连续音节,给它多个落点比给一个准。
 */
export function expandHotwords(terms: Iterable<string>): string[] {
  const out = new Set<string>()
  for (const raw of terms) {
    const t = raw.trim()
    if (t.length < 2) continue
    out.add(t)
    const lower = t.toLowerCase()
    if (lower !== t) out.add(lower)
    // 驼峰拆开:DeepEP → Deep EP,AllGather → All Gather
    const spaced = t.replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    if (spaced !== t) out.add(spaced)
  }
  return [...out]
}

/**
 * 播放与录音。
 *
 * 音量用 Web Audio 的 GainNode 而不是 `<audio>.volume` —— 后者上限是 1,只能调小不能放大。
 * 服务端已经把峰值归一化到 -1 dBFS,这里再留一档 0–200% 给你自己拧。
 */
export function useVoice() {
  const ctxRef = useRef<AudioContext | null>(null)
  const srcRef = useRef<AudioBufferSourceNode | null>(null)
  const recRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const stopSpeak = useCallback(() => {
    try {
      srcRef.current?.stop()
    } catch {
      // 已经停了
    }
    srcRef.current = null
  }, [])

  /** 合成并播放。返回音频时长(秒);出错抛异常由调用方处理。 */
  const speak = useCallback(
    async (text: string, cfg: VoiceConfig): Promise<number> => {
      const res = await fetch('/api/interview/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: text,
          model: cfg.ttsModel,
          voice: cfg.voice,
          instruct: cfg.instruct,
          speed: cfg.speed,
          temperature: cfg.temperature,
        }),
      })
      if (!res.ok) {
        const err = (await res.json().catch(() => ({}))) as { error?: string }
        throw new Error(err.error ?? `TTS ${res.status}`)
      }
      const buf = await res.arrayBuffer()
      const ctx = (ctxRef.current ??= new AudioContext())
      if (ctx.state === 'suspended') await ctx.resume()
      const decoded = await ctx.decodeAudioData(buf)

      stopSpeak()
      const src = ctx.createBufferSource()
      src.buffer = decoded
      const gain = ctx.createGain()
      gain.gain.value = cfg.volume
      src.connect(gain).connect(ctx.destination)
      src.start()
      srcRef.current = src
      return decoded.duration
    },
    [stopSpeak],
  )

  const startRecording = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    chunksRef.current = []
    const mr = new MediaRecorder(stream)
    mr.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data)
    recRef.current = mr
    mr.start()
  }, [])

  /** 停止录音并转写。hotwords 是这道题的术语,喂进去能显著提高召回。 */
  const stopAndTranscribe = useCallback(
    (cfg: VoiceConfig, hotwords: string[]): Promise<{ text: string; elapsedMs: number }> =>
      new Promise((resolve, reject) => {
        const mr = recRef.current
        if (!mr) return reject(new Error('没有在录音'))
        mr.onstop = async () => {
          mr.stream.getTracks().forEach((t) => t.stop())
          try {
            const blob = new Blob(chunksRef.current, { type: mr.mimeType })
            const fd = new FormData()
            fd.append('file', blob, 'answer.webm')
            fd.append('model', cfg.sttModel)
            if (hotwords.length) fd.append('hotwords', hotwords.join(','))
            const res = await fetch('/api/interview/transcribe', { method: 'POST', body: fd })
            const data = await res.json()
            if (data.error) return reject(new Error(data.error))
            resolve({ text: data.text ?? '', elapsedMs: data.elapsed_ms ?? 0 })
          } catch (e) {
            reject(e as Error)
          }
        }
        mr.stop()
        recRef.current = null
      }),
    [],
  )

  return { speak, stopSpeak, startRecording, stopAndTranscribe }
}
