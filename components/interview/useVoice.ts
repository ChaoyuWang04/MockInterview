'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'

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
  // 云端。同一段音频对比:云端标点齐全、术语全中;本地 Whisper 把「异步」听成「一步」。
  // 云端还吃得下全域热词表(本地 448 token 上下文会溢出)。没配 DASHSCOPE 时自动回落 whisper。
  sttModel: 'qwen3-asr-flash',
  enabled: true,
}

/** 语音设置存这个 key。改了设置就永久生效,刷新不丢。 */
const VOICE_STORAGE_KEY = 'interview.voice.v1'

/**
 * 收集热词:去重、去掉太短的,**保留原样写法**。
 *
 * 曾经这里会展开大小写与拆写变体(`DeepEP` → `deepep` / `Deep EP`),现已删除 ——
 * 对照实测(7.6s 音频,6 个驼峰术语)证明变体是负收益:
 *
 * | 热词表 | Whisper | Qwen3-ASR |
 * |---|---|---|
 * | 不给 | 4/6 | 6/6 |
 * | 只给规范写法 | **5/6** | 6/6 |
 * | 规范 + 变体 | **3/6** | 6/6 |
 *
 * 原因在机制上说得通:Whisper 把热词折进 `initial_prompt`,变体表更长、更像一份
 * 逗号分隔的词表,模型会去模仿那个风格 —— 实测直接把整句转成英文、丢掉大半内容。
 * 云端把热词放 system 的 Context 里,不受影响,但也没有增益。
 */
export function collectHotwords(terms: Iterable<string>): string[] {
  const out = new Set<string>()
  for (const raw of terms) {
    const t = raw.trim()
    if (t.length >= 2) out.add(t)
  }
  return [...out]
}

/**
 * 语音设置 + localStorage 持久化。
 *
 * 初值必须是 `DEFAULT_VOICE_CONFIG` 而不是直接读 localStorage —— 服务端渲染时没有
 * `window`,首屏两边对不上会 hydration 报错。所以挂载后再读、读完再允许写回。
 */
export function useVoiceConfig(): [VoiceConfig, Dispatch<SetStateAction<VoiceConfig>>] {
  const [config, setConfig] = useState<VoiceConfig>(DEFAULT_VOICE_CONFIG)
  const loaded = useRef(false)

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(VOICE_STORAGE_KEY)
      // 与默认值合并:以后给 VoiceConfig 加字段时,老的存档不会缺键
      if (raw) setConfig({ ...DEFAULT_VOICE_CONFIG, ...(JSON.parse(raw) as Partial<VoiceConfig>) })
    } catch {
      // 存档坏了就用默认值,不值得打断面试
    }
    loaded.current = true
  }, [])

  useEffect(() => {
    if (!loaded.current) return // 别用默认值盖掉还没读出来的存档
    try {
      window.localStorage.setItem(VOICE_STORAGE_KEY, JSON.stringify(config))
    } catch {
      // 隐私模式下 localStorage 会抛,设置退化成「本次有效」
    }
  }, [config])

  return [config, setConfig]
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
