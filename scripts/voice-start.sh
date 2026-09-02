#!/usr/bin/env bash
# 启动语音服务。读 .env.local 拿云端配置。
#
# ⚠️ 显式清掉代理变量:如果从带 HTTP_PROXY 的 shell 启动(比如 AI 助手的沙箱),
#    进程会继承它,导致云端 STT 连不上百炼 —— 而且表现是超时,不是报错。
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env.local ] && set -a && . ./.env.local && set +a
exec env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY \
  .venv-voice/bin/python scripts/voice-server.py
