#!/usr/bin/env bash
# 从 ModelScope 拉模型(HuggingFace 跨境限速到 200KB/s 时用它,实测快 80 倍)
#
#   ms-fetch.sh <namespace/repo> [目标目录]
#
# 不给目标目录时默认放 ~/.mlx-models/<repo名>。
# 特性:逐文件校验字节数、可重入(重跑跳过已下好的)、下载中用 .part 后缀。
set -uo pipefail

REPO="${1:?用法: ms-fetch.sh <namespace/repo> [目标目录]}"
DEST="${2:-$HOME/.mlx-models/$(basename "$REPO")}"
API="https://modelscope.cn/api/v1/models/$REPO"

mkdir -p "$DEST"
echo "仓库: $REPO"
echo "目标: $DEST"
echo

# ⚠️ 必须带 Recursive=True,并且用 Path 而不是 Name ——
# 不加的话只能拿到顶层文件,子目录整个丢掉,而且**不报错**:
# 踩过的坑是 Qwen3-TTS 的 speech_tokenizer/ 子目录(650MB)被静默漏掉,
# 模型能加载、一合成就报「Speech tokenizer not loaded」。
LIST=$(curl -sL --max-time 30 "$API/repo/files?Revision=master&Root=&Recursive=True" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
for f in d.get('Data', {}).get('Files', []):
    if f.get('Type') == 'tree':
        continue
    p = f.get('Path') or f.get('Name', '')
    if not p or p == '.gitattributes':
        continue
    print(f\"{p}\t{f.get('Size', 0)}\")
")

if [ -z "$LIST" ]; then
  echo "❌ 拿不到文件清单,中止"
  exit 1
fi

FAIL=0
while IFS=$'\t' read -r name size; do
  [ -z "$name" ] && continue
  out="$DEST/$name"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -f%z "$out")" = "$size" ]; then
    printf "  ✓ %-46s 已存在\n" "$name"
    continue
  fi
  printf "  ↓ %-46s %8.1f MB ... " "$name" "$(echo "$size" | awk '{print $1/1048576}')"
  if curl -sL --max-time 3600 --retry 3 --retry-delay 2 -o "$out.part" \
       "$API/repo?Revision=master&FilePath=$name"; then
    got=$(stat -f%z "$out.part" 2>/dev/null || echo 0)
    if [ "$got" = "$size" ]; then
      mv "$out.part" "$out"; echo "OK"
    else
      echo "❌ 大小不符(期望 $size,实得 $got)"; rm -f "$out.part"; FAIL=1
    fi
  else
    echo "❌ 下载失败"; rm -f "$out.part"; FAIL=1
  fi
done <<< "$LIST"

echo
if [ "$FAIL" = 0 ]; then
  echo "✅ 完成 · $(du -sh "$DEST" | cut -f1) · $DEST"
else
  echo "⚠️  有文件失败。本脚本可重入:重跑会跳过已下好的,只补失败的"
  exit 1
fi
