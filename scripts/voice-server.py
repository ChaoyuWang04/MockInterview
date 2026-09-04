#!/usr/bin/env python3
"""模拟面试的语音服务:TTS + STT,常驻不卸载模型。

    .venv-voice/bin/python scripts/voice-server.py

为什么要常驻:每次请求现加载模型要 2–10 秒,一轮面试就废了。
模型按需懒加载,加载后一直留在内存里。

接口走 OpenAI 兼容的形状,换云端只改 base URL:
    POST /v1/audio/speech           文字 → wav
    POST /v1/audio/transcriptions   wav  → 文字
    GET  /voices                    可用音色 / 可用 STT 后端
    GET  /health                    状态与内存占用
"""
import io
import os
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

MODELS_DIR = Path(os.environ.get("MLX_MODELS_DIR", Path.home() / ".mlx-models"))

# ---------------------------------------------------------------- 模型登记表
# 前端的下拉框直接读这里 —— 加一个模型只要在这加一行,不用改前端。
# 只留 VoiceDesign:预设音色那一版(CustomVoice)带口音,已弃用并删除。
TTS_MODELS = {
    "qwen3-tts-design": {
        "path": MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "label": "文字描述音色(VoiceDesign)",
        "kind": "design",
        "voices": [],
    },
}

#: 默认音色描述。显式写「标准普通话、无地方口音」—— 不写它模型会随机带腔调。
DEFAULT_INSTRUCT = "三十多岁的男性,标准普通话,播音腔,吐字清楚,语气平静克制不带感情,像在严肃地问问题"

#: 云端排第一 —— 它是默认。本地只留 Whisper 一个,断网/云端欠费时兜底。
#: (Fun-ASR Nano 曾是第三个本地后端,术语召回比 Whisper 差一档,已删除。)
STT_MODELS = {
    "qwen3-asr-flash": {
        "path": None,
        "label": "Qwen3-ASR Flash(云端,默认)",
        "kind": "dashscope",
    },
    "whisper-turbo": {
        "path": MODELS_DIR / "whisper-large-v3-turbo-asr-fp16",
        "label": "Whisper large-v3-turbo(本地)",
        "kind": "local",
    },
}

# 阿里云百炼。工作空间级 key 需要配套的 workspace 主机名,
# 用通用的 dashscope.aliyuncs.com 会 401。
DASHSCOPE_HOST = os.environ.get("DASHSCOPE_HOST", "").rstrip("/")
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

#: 全域热词表,由 `npm run interview:index` 从题库+知识库生成。
#: **只给云端 STT 用** —— 本地 Whisper 的解码上下文只有 448 token,灌全域会溢出并
#: 从前面截断,反而变差;云端灌 1421 词实测无影响,且按文档只计音频时长不计文本。
_DOMAIN_HOTWORDS_FILE = Path.cwd() / "interview" / "hotwords.txt"


def _domain_hotwords() -> list[str]:
    if not _DOMAIN_HOTWORDS_FILE.exists():
        return []
    return [x.strip() for x in _DOMAIN_HOTWORDS_FILE.read_text("utf8").splitlines() if x.strip()]

#: 转写临时文件的统一前缀。切段与 mp3 都从它派生(`-partN.wav` / `.mp3`),
#: 所以启动清扫只认这一个前缀就够。
STT_TMP_PREFIX = "interview-stt-"

DEFAULT_TTS = "qwen3-tts-design"
#: 云端。比本地 Whisper 准一档、快一倍,而且不吃本地 6.2 GB 内存;
#: 更关键的是它吃得下全域热词表(本地会溢出)。没配 DASHSCOPE 时前端会自动回落 whisper。
DEFAULT_STT = "qwen3-asr-flash"
DEFAULT_VOICE = ""  # VoiceDesign 没有预设音色,音色全靠 instruct
# 你反馈本地 TTS 说话偏慢,真人语速约是它的 1.3 倍
DEFAULT_SPEED = 1.2
#: 采样温度。模型默认 0.9,同一段描述两次生成音色差别很大;
#: 调低能显著收敛,但太低会让语调发平。0.4 是实测的折中点。
DEFAULT_TEMPERATURE = 0.1

#: 编解码器帧率(12Hz 模型),用来把「几个字」换算成「最多几个 token」
CODEC_HZ = 12
#: 每个中文字大约几秒语音(1.0 倍速实测约 0.15s),留 3 倍余量防误杀
SEC_PER_CHAR = 0.15
MAX_GEN_SECONDS = 120
#: 单句上限。超过就切开逐句合成 —— 长句会让模型生成跑飞:
#: 实测 41 字的题干出了 10–15 秒音频,中间大段静音、后半句根本没念;
#: 同样的内容切成 17 字的短句就完全正常。
MAX_CHARS_PER_CHUNK = 24
#: 句间停顿(秒)。太短像连读,太长像卡了
CHUNK_GAP_S = 0.18


def split_sentences(text: str, limit: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """按标点切句,再把过长的句子按逗号继续切。

    切分是为了稳定不是为了好听 —— 模型对长文本的生成会退化(丢句、拖静音),
    每段控制在 ~24 字以内就稳。切完逐段合成再拼接。
    """
    import re

    # 先按句末标点切,保留标点
    parts = [x for x in re.split(r"(?<=[。?!?!;;\n])", text) if x.strip()]
    out: list[str] = []
    for part in parts:
        p = part.strip()
        if len(p) <= limit:
            out.append(p)
            continue
        # 还是太长:按逗号/顿号再切
        subs = [x for x in re.split(r"(?<=[,,、::])", p) if x.strip()]
        buf = ""
        for sub in subs:
            if len(buf) + len(sub) <= limit:
                buf += sub
            else:
                if buf:
                    out.extend(_hard_split(buf.strip(), limit))
                buf = sub
        if buf.strip():
            out.extend(_hard_split(buf.strip(), limit))
    return out or [text]


def _hard_split(p: str, limit: int) -> list[str]:
    """没有标点可切的长句只能硬切。

    必须有这一步:`prefill 阶段和 decode 阶段对 KVCache 的处理有什么不同?` 这种
    40 字、中间一个逗号都没有的句子,只按标点切是切不开的 —— 而它正是生成跑飞的那批。
    优先在空格处断,断不了才按字数。
    """
    if len(p) <= limit:
        return [p]
    out, rest = [], p
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit + 1)
        if cut < limit // 2:
            cut = limit
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return [x for x in out if x]

_tts_cache: dict[str, object] = {}
_stt_cache: dict[str, object] = {}
_started = time.time()

app = FastAPI(title="interviewprep voice")


def _load_tts(key: str):
    if key not in _tts_cache:
        from mlx_audio.tts.utils import load_model

        spec = TTS_MODELS[key]
        if not spec["path"].exists():
            raise FileNotFoundError(f"模型不在:{spec['path']}(用 scripts/ms-fetch.sh 拉)")
        t = time.time()
        _tts_cache[key] = load_model(str(spec["path"]))
        print(f"[tts] 加载 {key} 用时 {time.time()-t:.1f}s", flush=True)
    return _tts_cache[key]


def _load_stt(key: str):
    if key not in _stt_cache:
        from mlx_audio.stt.utils import load_model

        spec = STT_MODELS[key]
        if not spec["path"].exists():
            raise FileNotFoundError(f"模型不在:{spec['path']}(用 scripts/ms-fetch.sh 拉)")
        t = time.time()
        _stt_cache[key] = load_model(str(spec["path"]))
        print(f"[stt] 加载 {key} 用时 {time.time()-t:.1f}s", flush=True)
    return _stt_cache[key]


def _stt_ready(spec: dict) -> bool:
    if spec["kind"] == "dashscope":
        return bool(DASHSCOPE_HOST and DASHSCOPE_KEY)
    return spec["path"] is not None and spec["path"].exists()


#: 百炼单次的时长上限,**实测正好 300 秒整**:300s 过、302s 就报 `The audio is too long`。
#: 这里取 290 留余量 —— 卡在 300 上会被时长测量的零点几秒差异咬到
#: (我们用 wave 读帧数算,百炼用它自己的解码器算,两边不保证同一个数)。
DASHSCOPE_MAX_SECONDS = 290
#: 切段长度。面试回答很少有超过四分钟还不换气的,240 让绝大多数长回答只切两段。
CHUNK_SECONDS = 240


def _wav_seconds(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def _silences(p: Path) -> list[tuple[float, float]]:
    """用 ffmpeg 找静音区间,给切点用。找不到就返回空,调用方硬切。"""
    import re
    import subprocess

    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(p), "-af",
             "silencedetect=noise=-35dB:d=0.35", "-f", "null", "-"],
            capture_output=True, timeout=180, text=True,
        )
    except Exception:  # noqa: BLE001
        return []
    log = r.stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    return list(zip(starts, ends))


def _cut_points(p: Path, total: float) -> list[float]:
    """切点:每 CHUNK_SECONDS 一刀,**尽量落在静音里**,免得把词切成两半。

    目标点附近 ±30s 内有静音就切在静音正中;没有就硬切在目标点 ——
    硬切只是让边界那个词可能听错一次,不会让整段转写失败。
    """
    sil = _silences(p)
    points: list[float] = []
    t = CHUNK_SECONDS
    while t < total - 20:  # 尾巴不足 20s 就并进上一段
        best = min(
            (s for s in sil if abs((s[0] + s[1]) / 2 - t) <= 30),
            key=lambda s: abs((s[0] + s[1]) / 2 - t),
            default=None,
        )
        cut = (best[0] + best[1]) / 2 if best else t
        if points and cut - points[-1] < 30:  # 别切出太碎的段
            cut = t
        points.append(cut)
        t = cut + CHUNK_SECONDS
    return points


def _slice_wav(src: Path, start: float, end: float, idx: int) -> Path:
    import subprocess

    out = src.with_name(f"{src.stem}-part{idx}.wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-ac", "1", "-ar", "16000", "-y", str(out)],
        capture_output=True, timeout=180, check=True,
    )
    return out


def _dashscope_transcribe(wav_path: Path, hotwords: list[str]) -> str:
    """云端转写。超过单次时长上限就切段并发转,再按序拼接。"""
    total = _wav_seconds(wav_path)
    if total <= DASHSCOPE_MAX_SECONDS:
        return _dashscope_once(wav_path, hotwords)

    from concurrent.futures import ThreadPoolExecutor

    bounds = [0.0, *_cut_points(wav_path, total), total]
    parts: list[Path] = []
    try:
        for i in range(len(bounds) - 1):
            parts.append(_slice_wav(wav_path, bounds[i], bounds[i + 1], i))
        print(f"[stt] 音频 {total:.0f}s 超过单次上限,切成 {len(parts)} 段并发转写", flush=True)
        # map 保序 —— 拼接顺序不能乱。三路并发,再多没必要也容易触发限流
        with ThreadPoolExecutor(max_workers=3) as ex:
            texts = list(ex.map(lambda p: _dashscope_once(p, hotwords), parts))
    finally:
        for p in parts:
            p.unlink(missing_ok=True)
    return " ".join(t.strip() for t in texts if t.strip())


def _dashscope_once(wav_path: Path, hotwords: list[str]) -> str:
    """走百炼的 qwen3-asr-flash。

    两个关键点:
    ① 音频用 base64 内联(data URI)—— 官方 filetrans 那条要求可公网访问的 URL,本地文件用不了
    ② 热词塞进 system 的 Context —— 实测术语召回 15/20 → **18/20**,而且不增加耗时。
       `CUDA Graph` 这种词,本地模型全军覆没(Kilographs / Kill the Graph),带 Context 才认得出。
    """
    import json, urllib.error, urllib.request

    context = ""
    if hotwords:
        context = (
            "这是一段大模型推理与训练基础设施方向的技术面试录音,可能出现的专有名词:"
            + ", ".join(hotwords)
        )

    audio, mime = _inline_audio(wav_path)
    body = {
        "model": "qwen3-asr-flash",
        "input": {"messages": [
            {"role": "system", "content": [{"text": context}]},
            {"role": "user", "content": [{"audio": f"data:{mime};base64,{audio}"}]}]},
        "parameters": {"asr_options": {"enable_lid": True, "enable_itn": False}},
    }
    req = urllib.request.Request(
        f"{DASHSCOPE_HOST}/api/v1/services/aigc/multimodal-generation/generation",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {DASHSCOPE_KEY}"},
    )
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        # ⚠️ 一定要把 body 读出来。不读的话异常文本只有「HTTP Error 400: Bad Request」,
        # 而百炼真正的原因写在 body 里(踩过:`Multimodal file size is too large`
        # 被这层吞掉,只能靠二分音频长度反推)。
        raise RuntimeError(f"百炼 {e.code}:{e.read().decode('utf8', 'replace')[:300]}") from None
    return d["output"]["choices"][0]["message"]["content"][0]["text"]


#: 百炼内联音频的体积上限(实测:9.4 MB 过、14 MB 不过,报
#: `InvalidParameter: Multimodal file size is too large`)。留一成余量。
DASHSCOPE_MAX_INLINE_BYTES = 9_500_000


def _inline_audio(wav_path: Path) -> tuple[str, str]:
    """把音频编码成 base64 内联串。**先压 mp3 再传。**

    原来直接传未压缩 WAV:16kHz 单声道 = 32 KB/秒,**5 分半就撞 10 MB 上限**,
    而且表现是一句看不懂的 400。压成 32k 单声道 mp3 之后同样的上限约等于 40 分钟,
    实测对识别准确率没有影响(ASR 本来就重采样到 16k)。

    ffmpeg 不在或转码失败时回落原 WAV —— 短音频照样能用,不因为压缩这一步整个挂掉。
    """
    import base64
    import subprocess

    mp3 = wav_path.with_suffix(".mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(wav_path),
             "-ac", "1", "-ar", "16000", "-b:a", "32k", "-y", str(mp3)],
            capture_output=True, timeout=120, check=True,
        )
        raw, mime = mp3.read_bytes(), "audio/mp3"
    except Exception:  # noqa: BLE001
        raw, mime = wav_path.read_bytes(), "audio/wav"
    finally:
        mp3.unlink(missing_ok=True)

    if len(raw) > DASHSCOPE_MAX_INLINE_BYTES:
        mins = len(raw) / 4000 / 60  # 32kbps ≈ 4 KB/s
        raise RuntimeError(
            f"这段录音压缩后还有 {len(raw)/1e6:.1f} MB(约 {mins:.0f} 分钟),"
            f"超过云端单次 {DASHSCOPE_MAX_INLINE_BYTES/1e6:.1f} MB 的上限。"
            f"换本地 Whisper 后端可以转(没有时长限制,只是慢),或者分两次答。"
        )
    return base64.b64encode(raw).decode(), mime


# ---------------------------------------------------------------- 读音修正
#
# TTS 按字面读,而技术术语的写法和读法经常对不上:
#   all2all  → 念成「all 二 all」,应该是 all to all
#   deepep   → 全小写当一个词念,应该是 Deep E P
#   vLLM     → 念成「v-l-l-m」含糊,拆成字母更清楚
#
# 所以合成前先做一遍文本改写。**只影响念出来的声音,不影响判卷**——
# 判卷用的是原文,这里改的是喂给 TTS 的那份拷贝。
PRONUNCIATION = {
    # ── 只有一类真需要改:数字当介词用,TTS 会念成中文数字(「all 二 all」)──
    "all2all": "all to all",
    "a2a": "A to A",
    "s2s": "S to S",
    "t2i": "T to I",
    "i2v": "I to V",
    "t2v": "T to V",
    "seq2seq": "seq to seq",
    # ── 其余只做「大小写规范化」,**不插空格** ──
    # 模型已经学会按大写字母切词、并把连续大写逐字母读,所以 `DeepEP` 就够了;
    # 拆成 `Deep E P` 反而把句子打碎,让生成更不稳(实测同一句从 5.6s 拖到 8.2s)。
    "deepep": "DeepEP",
    "kvcache": "KVCache",
    "kv cache": "KVCache",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "nccl": "NCCL",
    "rdma": "RDMA",
    "nvlink": "NVLink",
    "cudagraph": "CUDAGraph",
    "cuda graph": "CUDAGraph",
    "pagedattention": "PagedAttention",
    "flashattention": "FlashAttention",
    "radixattention": "RadixAttention",
    "prefixgrouper": "PrefixGrouper",
    "roofline": "Roofline",
    "chunked prefill": "chunked prefill",
    "ngram": "NGram",
}

# 规则很简单:**能靠大小写解决的就别插空格**。
# 听着不对就往上面加一条(左边全小写、右边写成你希望它念的样子),
# 加完重跑 `npm run interview:tts` 让预渲染跟上。

#: 按长度倒序,先替换长词 —— 否则 "ep" 会先把 "deepep" 里的 ep 换掉
_PRON_KEYS = sorted(PRONUNCIATION, key=len, reverse=True)


def normalize_for_tts(text: str) -> str:
    """把技术术语改写成「按这样念」的形式。整词匹配,不碰中文。"""
    import re

    out = text
    for k in _PRON_KEYS:
        # 词边界用「非字母数字」判定,这样 KV cache 里的 kv 能命中,deepep 里的 ep 不会
        out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![A-Za-z0-9])",
                     PRONUNCIATION[k], out, flags=re.IGNORECASE)
    return out


#: 归一化目标(dBFS)。-1 留一点余量防削波,是语音素材的常规做法。
PEAK_TARGET_DB = -1.0


def _looks_broken(a: np.ndarray, n_chars: int, sr: int = 24000) -> bool:
    """判断这段生成是不是跑飞了。

    生成是随机采样,切分只降低概率、消不掉 —— 所以还要在输出侧自检。
    两个判据:静音占比过高(模型念一半就停了拖空白),或时长远超文本应有的长度。
    """
    if a.size < sr // 4:
        return True
    win = max(1, sr // 4)
    n = a.size // win
    if n >= 2:
        e = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(axis=1))
        if (e < 0.01).mean() > 0.4:
            return True
    return a.size / sr > max(6.0, n_chars * SEC_PER_CHAR * 2.5)


def _trim_silence(a: np.ndarray, thresh: float = 0.01, sr: int = 24000) -> np.ndarray:
    """掐掉首尾静音。模型经常在句末拖几秒空白,逐段拼接前必须去掉,
    否则一道题下来能拖出十几秒的死气。"""
    if a.size == 0:
        return a
    win = max(1, sr // 100)
    n = a.size // win
    if n < 2:
        return a
    energy = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(axis=1))
    loud = np.nonzero(energy > thresh)[0]
    if loud.size == 0:
        return a[: sr // 4]  # 整段都是静音,只留一点
    lo = max(0, loud[0] - 2) * win
    hi = min(n, loud[-1] + 3) * win
    return a[lo:hi]


def _normalize(samples: np.ndarray) -> np.ndarray:
    """把峰值拉到 -1 dBFS。

    Qwen3-TTS 的原始输出偏轻:实测峰值只有 0.421(-7.5 dBFS)、RMS -24 dBFS,
    比常规语音素材低一大截,在笔记本外放上明显偏小。这里把那 7dB 余量用掉,
    响度约 ×2.1,而且是纯线性增益,不改音色也不引入失真。
    """
    peak = float(np.abs(samples).max())
    if peak < 1e-6:
        return samples
    target = 10.0 ** (PEAK_TARGET_DB / 20.0)
    return samples * (target / peak)


def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1,1] → 16-bit PCM wav。浏览器 <audio> 直接能播。"""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _retime(wav_bytes: bytes, speed: float) -> bytes:
    """变速不变调。

    Qwen3-TTS 的 `speed` 参数是**空的** —— 源码里明写 "not directly supported yet",
    传进去只会禁用一条批处理路径,音频时长一点不变(实测 1.0x 得 9.92s、1.3x 反而 12.32s)。
    所以速度由我们在输出侧做:走 ffmpeg 的 atempo 滤镜,变速不变调,
    10 秒音频约 20ms,而且不用引入 librosa 那一坨依赖。
    """
    if abs(speed - 1.0) < 0.01:
        return wav_bytes
    # atempo 单次只接受 0.5–2.0,超出范围要串联
    factors, remaining = [], speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    chain = ",".join(f"atempo={f:.4f}" for f in factors)

    import subprocess

    try:
        # ⚠️ 必须输出裸 PCM 再自己封头。让 ffmpeg 直接 `-f wav pipe:1` 的话,
        # 管道不可 seek,它回填不了 RIFF 的长度字段,写成 0xFFFFFFFF ——
        # 播放器会把时长读成 89478 秒(踩过)。
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-filter:a", chain, "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"],
            input=wav_bytes, capture_output=True, timeout=30, check=True,
        )
        if not out.stdout:
            return wav_bytes
        with wave.open(io.BytesIO(wav_bytes)) as src:
            sr, ch, sw = src.getframerate(), src.getnchannels(), src.getsampwidth()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(sw)
            w.setframerate(sr)
            w.writeframes(out.stdout)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        print(f"[tts] 变速失败,回退原速:{e}", flush=True)
        return wav_bytes


@app.get("/health")
def health():
    import mlx.core as mx

    return {
        "ok": True,
        "uptime_s": round(time.time() - _started, 1),
        "tts_loaded": list(_tts_cache),
        "stt_loaded": list(_stt_cache),
        "mem_gb": round(mx.get_active_memory() / 1e9, 2),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


@app.get("/voices")
def voices():
    """前端下拉框的数据源。只列磁盘上真实存在的模型。"""
    return {
        "tts": [
            {"id": k, "label": v["label"], "kind": v["kind"], "voices": v["voices"],
             "ready": v["path"].exists()}
            for k, v in TTS_MODELS.items()
        ],
        "stt": [
            {"id": k, "label": v["label"], "kind": v["kind"], "ready": _stt_ready(v)}
            for k, v in STT_MODELS.items()
        ],
        "defaults": {"tts": DEFAULT_TTS, "stt": DEFAULT_STT, "voice": DEFAULT_VOICE,
                     "speed": DEFAULT_SPEED, "instruct": DEFAULT_INSTRUCT,
                     "temperature": DEFAULT_TEMPERATURE},
    }


@app.post("/v1/audio/speech")
def speech(payload: dict):
    """⚠️ 这里必须是 `def` 而不是 `async def`。
    模型推理是同步阻塞的:写成 `async def` 会占着事件循环不放,
    一个慢请求就让**整个服务**失去响应(踩过:一次失控生成跑了 149 秒,
    期间连 /health 都超时)。写成 `def`,FastAPI 会把它丢进线程池。"""
    text = (payload.get("input") or "").strip()
    if not text:
        return JSONResponse({"error": "input 是空的"}, status_code=400)
    model_key = payload.get("model") or DEFAULT_TTS
    voice = payload.get("voice") or DEFAULT_VOICE
    instruct = (payload.get("instruct") or DEFAULT_INSTRUCT).strip()
    speed = float(payload.get("speed") or DEFAULT_SPEED)
    temperature = float(payload.get("temperature") or DEFAULT_TEMPERATURE)
    if model_key not in TTS_MODELS:
        return JSONResponse({"error": f"没有这个 TTS 模型:{model_key}"}, status_code=400)
    is_design = TTS_MODELS[model_key]["kind"] == "design"

    import mlx.core as mx

    t = time.time()
    try:
        model = _load_tts(model_key)
        spoken = normalize_for_tts(text) if payload.get("normalize", True) else text
        # 注意:speed 不传给模型 —— 它不认(见 _retime 的说明),由输出侧统一处理。
        # VoiceDesign 用 instruct 描述音色(没有预设音色);CustomVoice 用 voice 选预设。
        gen_kw = {"instruct": instruct} if is_design else {"voice": voice}
        chunks = split_sentences(spoken)
        gap = np.zeros(int(24000 * CHUNK_GAP_S), dtype=np.float32)
        pieces: list[np.ndarray] = []
        budget_s = 0.0
        for ci, chunk in enumerate(chunks):
            # 失控生成保护:按本段长度算 token 上限,留 3 倍余量。
            # 模型偶尔会在映射不了的写法上打转,一路跑到默认的 max_tokens=4096
            # (≈341 秒音频),实测一次吃了 149 秒和 24.7GB 内存。
            b = min(MAX_GEN_SECONDS, max(6.0, len(chunk) * SEC_PER_CHAR * 3))
            budget_s += b
            arr_c = None
            # 最多试 3 次:生成是随机的,跑飞的那次重来一般就好了
            for attempt in range(3):
                segs = list(model.generate(
                    text=chunk, language="chinese", verbose=False,
                    temperature=temperature, max_tokens=int(b * CODEC_HZ), **gen_kw))
                au = segs[0].audio if len(segs) == 1 else mx.concatenate([s_.audio for s_ in segs])
                cand = _trim_silence(np.array(au, copy=False).reshape(-1))
                if not _looks_broken(cand, len(chunk)):
                    arr_c = cand
                    break
                print(f"[tts] 第{attempt+1}次生成异常,重试:{chunk[:30]}", flush=True)
                arr_c = cand
            pieces.append(arr_c)
            if ci < len(chunks) - 1:
                pieces.append(gap)
        segs = None
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    arr = _normalize(np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32))
    sr = 24000
    raw_dur = len(arr) / sr
    wav = _retime(_to_wav_bytes(arr, sr), speed)
    dur = raw_dur / speed
    elapsed = time.time() - t
    print(
        f"[tts] {instruct[:16] + '…' if is_design else voice}@{speed}x {len(text)}字 → {raw_dur:.1f}s→{dur:.1f}s "
        f"用时{elapsed:.2f}s RTF{elapsed/dur:.2f}",
        flush=True,
    )
    if spoken != text:
        print(f"[tts] 读音修正: {spoken[:90]}", flush=True)
    if raw_dur > budget_s * 0.95:
        print(
            f"[tts] ⚠️ 生成疑似失控:{len(spoken)}字 出了 {raw_dur:.0f}s 音频(上限 {budget_s:.0f}s)。"
            f"多半是文本里有模型映射不了的写法 —— 开「读音修正」通常能避免。",
            flush=True,
        )
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Elapsed-Ms": str(round(elapsed * 1000)),
            "X-Audio-Seconds": f"{dur:.2f}",
            "X-Rtf": f"{elapsed/dur:.2f}",
            "X-Spoken": __import__("urllib.parse", fromlist=["quote"]).quote(spoken[:400]),
        },
    )


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_STT),
    hotwords: Optional[str] = Form(None),
):
    """hotwords 用逗号分隔。系统知道当前题会出现哪些术语,喂进来能提高召回。

    **只传规范写法,不要传大小写/拆写变体。** 实测(7.6s 音频,6 个驼峰术语):
    Whisper 无热词 4/6 · 只给规范写法 5/6 · 规范+变体 **3/6** —— 变体表更长更像
    一份词表,会放大 initial_prompt 的模仿效应,把输出带跑偏(整句转英文、丢内容)。
    云端三种配置都是 6/6,变体对它没有增益。"""
    if model not in STT_MODELS:
        return JSONResponse({"error": f"没有这个 STT 模型:{model}"}, status_code=400)

    raw = file.file.read()
    # 浏览器 MediaRecorder 给的是 webm/opus(Chrome)或 mp4/aac(Safari),
    # ASR 模型只吃 wav。统一用 ffmpeg 转成 16k 单声道 —— 顺带做重采样,
    # 省得每个后端各自处理采样率。
    # 切段与 mp3 都是从这个名字派生的(`-partN.wav` / `.mp3`),所以清扫认前缀就够
    tmp = Path("/tmp") / f"{STT_TMP_PREFIX}{os.getpid()}-{int(time.time()*1000)}.wav"
    import subprocess

    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-ac", "1", "-ar", "16000", "-f", "wav", "-y", str(tmp)],
            input=raw, capture_output=True, timeout=60, check=True,
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"音频转码失败:{e}"}, status_code=400)
    terms = [x.strip() for x in (hotwords or "").split(",") if x.strip()]
    t = time.time()
    try:
        if STT_MODELS[model]["kind"] == "dashscope":
            if not _stt_ready(STT_MODELS[model]):
                return JSONResponse(
                    {"error": "云端 STT 没配好:需要 DASHSCOPE_HOST 和 DASHSCOPE_API_KEY"},
                    status_code=400,
                )
            # 本题热词放前面(最相关),全域词表补在后面。
            # 云端不溢出,所以能同时吃下两者;本地后端只用本题热词。
            merged = terms + [w for w in _domain_hotwords() if w not in set(terms)]
            text = _dashscope_transcribe(tmp, merged).strip()
        else:
            m = _load_stt(model)
            kw = {"hotwords": terms} if terms else {}
            try:
                res = m.generate(str(tmp), **kw)
            except TypeError:
                res = m.generate(str(tmp))  # 该后端不认 hotwords
            text = (getattr(res, "text", None) or str(res)).strip()
    except RuntimeError as e:
        # 我们自己抛的、写给人看的原因(体积超限、百炼的原始报错)—— 原样透出去,
        # 别再套一层类型名把它变成天书
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    finally:
        tmp.unlink(missing_ok=True)

    elapsed = time.time() - t
    print(f"[stt] {model} {len(raw)/1024:.0f}KB 用时{elapsed:.2f}s → {text[:60]}", flush=True)
    return {"text": text, "model": model, "elapsed_ms": round(elapsed * 1000)}


def _sweep_stt_tmp() -> int:
    """启动时清掉上次残留的转写临时文件。

    正常路径上每个临时文件都有 `finally` 兜底删除,但进程被 SIGKILL / 断电时
    `finally` 不会执行。那种情况下几百 MB 的 wav 会一直躺在 /tmp 里 ——
    不报错、不影响功能,只是悄悄占空间,所以放在启动时扫一遍。
    """
    n = 0
    for p in Path("/tmp").glob(f"{STT_TMP_PREFIX}*"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


if __name__ == "__main__":
    import uvicorn

    left = _sweep_stt_tmp()
    if left:
        print(f"清理了 {left} 个上次残留的转写临时文件", flush=True)
    port = int(os.environ.get("INTERVIEW_VOICE_PORT", "8700"))
    print(f"语音服务 → http://127.0.0.1:{port}  (模型目录 {MODELS_DIR})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
