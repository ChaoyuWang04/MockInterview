#!/usr/bin/env bash
# 语音控件端到端审计:bash scripts/voice-audit.sh
#
# 逐个验证前端每个控件改了值之后,后端行为是否真的跟着变。
# 存在的理由:这条链路有三跳(浏览器 → Next.js 代理 → voice-server),
# 中间任何一跳漏传参数都**不报错**,只是行为对不上 —— 已经踩过三次:
#   · 代理漏传 instruct   → 音色描述改了不生效,一直是默认男声
#   · 代理漏传 normalize  → 读音修正开关是死的
#   · 代理漏传 X-Spoken   → 前端「实际念的是…」永远不显示
#
# 前提:npm run dev(:3001)与 scripts/voice-start.sh(:8700)都在跑。
API=http://localhost:3001/api/interview
pass=0; fail=0
ok()   { printf "  ✅ %-22s %s\n" "$1" "$2"; pass=$((pass+1)); }
bad()  { printf "  ❌ %-22s %s\n" "$1" "$2"; fail=$((fail+1)); }

tts() { # $1=json body → 输出 "音频秒数|念的文本"
  curl -s --max-time 200 -D /tmp/a.h -o /dev/null "$API/voice" \
    -H "Content-Type: application/json" -d "$1" >/dev/null
  sec=$(grep -i x-audio-seconds /tmp/a.h | tr -d '\r' | cut -d' ' -f2)
  sp=$(grep -i x-spoken /tmp/a.h | tr -d '\r' | cut -d' ' -f2)
  echo "$sec|$sp"
}

TXT="moe 的 dispatch 有哪些方案?all2all、deepep 通信量分别是多少?"

echo "── TTS 侧 ──"

# 1. 语速 —— 必须量真实文件时长,而且同一段文本各跑 3 次取最小,
#    因为每次生成的原始音频长度本身会抖(采样温度),单次对比不成立。
dur_of() { python3 -c "
import wave,sys
w=wave.open(sys.argv[1]); print(f'{w.getnframes()/w.getframerate():.2f}')" "$1"; }
best() { # $1=speed → 3 次里最短的真实时长
  m=999
  for i in 1 2 3; do
    curl -s --max-time 200 -o /tmp/sp.wav "$API/voice" -H "Content-Type: application/json" \
      -d "{\"input\":\"$TXT\",\"model\":\"qwen3-tts-design\",\"speed\":$1,\"temperature\":0.3,\"normalize\":true}" >/dev/null
    d=$(dur_of /tmp/sp.wav)
    m=$(awk -v a="$m" -v b="$d" 'BEGIN{print (b<a)?b:a}')
  done
  echo "$m"
}
a=$(best 1.0); b=$(best 1.6)
awk -v a="$a" -v b="$b" 'BEGIN{exit !(a>b*1.25)}' \
  && ok "语速 speed" "1.0×→${a}s  1.6×→${b}s(真实文件时长,变短)" \
  || bad "语速 speed" "1.0×→${a}s  1.6×→${b}s(应约 1.6 倍差)"

# 2. 读音修正
n0=$(tts "{\"input\":\"$TXT\",\"model\":\"qwen3-tts-design\",\"speed\":1.3,\"normalize\":false}" | cut -d'|' -f2)
n1=$(tts "{\"input\":\"$TXT\",\"model\":\"qwen3-tts-design\",\"speed\":1.3,\"normalize\":true}"  | cut -d'|' -f2)
[ "$n0" != "$n1" ] && ok "读音修正 normalize" "关/开 念的文本不同" \
                   || bad "读音修正 normalize" "两者相同 → 参数没生效"

# 3. 音色描述(男 vs 女,比音频长度 + 确认无报错)
m=$(tts "{\"input\":\"什么是算力利用率?\",\"model\":\"qwen3-tts-design\",\"instruct\":\"六十岁老年男性,声音沙哑低沉,语速很慢\",\"speed\":1.0,\"temperature\":0.3}" | cut -d'|' -f1)
f=$(tts "{\"input\":\"什么是算力利用率?\",\"model\":\"qwen3-tts-design\",\"instruct\":\"二十岁年轻女性,声音清脆明亮,语速很快\",\"speed\":1.0,\"temperature\":0.3}" | cut -d'|' -f1)
[ -n "$m" ] && [ -n "$f" ] && [ "$m" != "$f" ] \
  && ok "音色描述 instruct" "老男${m}s vs 少女${f}s(不同)" \
  || bad "音色描述 instruct" "老男${m}s vs 少女${f}s(相同=可能没生效)"

# 4. 预设音色
p1=$(tts "{\"input\":\"什么是算力利用率?\",\"model\":\"qwen3-tts-1.7b\",\"voice\":\"eric\",\"speed\":1.0,\"temperature\":0.3}" | cut -d'|' -f1)
p2=$(tts "{\"input\":\"什么是算力利用率?\",\"model\":\"qwen3-tts-1.7b\",\"voice\":\"vivian\",\"speed\":1.0,\"temperature\":0.3}" | cut -d'|' -f1)
[ -n "$p1" ] && [ -n "$p2" ] && ok "预设音色 voice" "eric ${p1}s / vivian ${p2}s" \
                             || bad "预设音色 voice" "有请求失败"

# 5. 模型切换
mm=$(tts "{\"input\":\"测试\",\"model\":\"qwen3-tts-1.7b\",\"voice\":\"eric\",\"speed\":1.0}" | cut -d'|' -f1)
md=$(tts "{\"input\":\"测试\",\"model\":\"qwen3-tts-design\",\"instruct\":\"男性\",\"speed\":1.0}" | cut -d'|' -f1)
[ -n "$mm" ] && [ -n "$md" ] && ok "音色来源 ttsModel" "两个模型都能出声" \
                             || bad "音色来源 ttsModel" "preset=${mm} design=${md}"

echo "── STT 侧 ──"
WAV=/tmp/tts-out/02-术语密集.wav
stt() { curl -s --max-time 240 -X POST "$API/transcribe" -F "file=@$WAV" -F "model=$1" ${2:+-F "hotwords=$2"} ; }

for m in fun-asr-nano whisper-turbo qwen3-asr-flash; do
  t=$(stt "$m" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('text','ERR')[:40])" 2>/dev/null)
  [ -n "$t" ] && [ "$t" != "ERR" ] && ok "STT 模型 $m" "$t" || bad "STT 模型 $m" "无输出"
done

h0=$(stt whisper-turbo | python3 -c "import sys,json;print(json.load(sys.stdin).get('text','')[:60])")
h1=$(stt whisper-turbo "MoE,DeepEP,deepep,Deep EP,all2all,AllGather,AllReduce" | python3 -c "import sys,json;print(json.load(sys.stdin).get('text','')[:60])")
[ "$h0" != "$h1" ] && ok "热词 hotwords" "带/不带 结果不同" || bad "热词 hotwords" "结果相同 → 可能没生效"

echo
echo "  通过 $pass · 失败 $fail"
