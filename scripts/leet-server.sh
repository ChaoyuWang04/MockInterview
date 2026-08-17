#!/usr/bin/env bash
# 刷题系统常驻服务(生产模式)
#
# 用法:在 ~/.zshrc 里 source 本文件,即可使用:
#   leet-start    启动常驻服务(必要时自动构建)
#   leet-stop     停止
#   leet-restart  重启
#   leet-status   查看状态(PID/内存/CPU/是否已脱离终端)
#   leet-rebuild  改过代码后重新构建并重启
#   leet-logs     跟踪日志
#   leet-open     浏览器打开
#
# 关键:服务由 scripts/leet-daemon.py 以 double-fork + setsid 启动,
#      进程无控制终端、父进程为 init —— 关掉终端窗口或退出 Terminal.app 都不受影响。
#
# 说明:改 markdown 内容(题目/笔记/知识库/解读)无需重启,刷新页面即可;
#      只有改 app/ components/ lib/ 等代码才需要 leet-rebuild。

LEETPREP_DIR="${LEETPREP_DIR:-$HOME/Desktop/1Project/interviewprep}"
LEETPREP_PORT="${LEETPREP_PORT:-3000}"

_leet_paths() {
  LEET_RUN_DIR="$LEETPREP_DIR/.leet"
  LEET_PID_FILE="$LEET_RUN_DIR/server.pid"
  LEET_LOG_FILE="$LEET_RUN_DIR/server.log"
  mkdir -p "$LEET_RUN_DIR"
}

_leet_pid() {
  _leet_paths
  [ -f "$LEET_PID_FILE" ] || return 1
  local pid
  pid=$(tr -dc '0-9' <"$LEET_PID_FILE" 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

leet-start() {
  _leet_paths
  local pid
  if pid=$(_leet_pid); then
    echo "✓ 已在运行(PID $pid)→ http://localhost:$LEETPREP_PORT"
    return 0
  fi
  if lsof -ti:"$LEETPREP_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "✗ 端口 $LEETPREP_PORT 被别的进程占用:$(lsof -ti:"$LEETPREP_PORT" -sTCP:LISTEN | tr '\n' ' ')"
    echo "  换端口:LEETPREP_PORT=3002 leet-start"
    return 1
  fi

  cd "$LEETPREP_DIR" || { echo "✗ 找不到目录 $LEETPREP_DIR"; return 1; }
  # 生产构建独立放在 .next-prod,不会被 npm run dev(端口 3001)冲掉
  if [ ! -f "$LEETPREP_DIR/.next-prod/BUILD_ID" ]; then
    echo "· 没有生产构建,先构建(约 30-60 秒)…"
    npm run build >"$LEET_LOG_FILE" 2>&1 || { echo "✗ 构建失败,看日志:leet-logs"; return 1; }
  fi

  LEET_NODE_BIN="$(command -v node)" python3 "$LEETPREP_DIR/scripts/leet-daemon.py" \
    "$LEETPREP_DIR" "$LEETPREP_PORT" "$LEET_LOG_FILE" "$LEET_PID_FILE" || {
      echo "✗ 启动失败,看日志:leet-logs"; return 1; }

  local i
  for i in $(seq 1 40); do
    if curl -fsS -o /dev/null "http://localhost:$LEETPREP_PORT" 2>/dev/null; then
      echo "✓ 已启动(PID $(_leet_pid))→ http://localhost:$LEETPREP_PORT"
      echo "  已脱离终端:可以直接关窗口/退出 Terminal,服务照常运行;停止用 leet-stop"
      return 0
    fi
    sleep 0.5
  done
  echo "✗ 启动超时,看日志:leet-logs"
  return 1
}

leet-stop() {
  _leet_paths
  local pid
  if pid=$(_leet_pid); then
    kill "$pid" 2>/dev/null
    local i
    for i in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
    rm -f "$LEET_PID_FILE"
    echo "✓ 已停止(PID $pid)"
  else
    rm -f "$LEET_PID_FILE"
    # 兜底:PID 文件丢了但端口还占着
    local orphan
    orphan=$(lsof -ti:"$LEETPREP_PORT" -sTCP:LISTEN 2>/dev/null)
    if [ -n "$orphan" ]; then
      echo "$orphan" | xargs kill 2>/dev/null
      echo "✓ 已清理端口 $LEETPREP_PORT 上的残留进程"
    else
      echo "· 服务本来就没在运行"
    fi
  fi
}

leet-restart() { leet-stop; leet-start; }

leet-status() {
  _leet_paths
  local pid
  if pid=$(_leet_pid); then
    local mem cpu tty
    mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f MB", $1/1024}')
    cpu=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')
    tty=$(ps -o tty= -p "$pid" 2>/dev/null | tr -d ' ')
    echo "✓ 运行中 · PID $pid · 内存 ${mem:-?} · CPU ${cpu:-?}%"
    echo "  http://localhost:$LEETPREP_PORT · 控制终端:${tty:-??}(?? = 已脱离终端)"
  else
    echo "· 未运行(leet-start 启动)"
  fi
}

leet-rebuild() {
  _leet_paths
  cd "$LEETPREP_DIR" || return 1
  echo "· 重新构建…"
  npm run build >"$LEET_LOG_FILE" 2>&1 || { echo "✗ 构建失败,看日志:leet-logs"; return 1; }
  echo "✓ 构建完成"
  leet-restart
}

leet-logs() { _leet_paths; tail -f "$LEET_LOG_FILE"; }

leet-open() { open "http://localhost:$LEETPREP_PORT"; }

# 直接执行(而非 source)时按子命令分发
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ] && [ -n "${1:-}" ]; then
  case "$1" in
    start) leet-start ;;
    stop) leet-stop ;;
    restart) leet-restart ;;
    status) leet-status ;;
    rebuild) leet-rebuild ;;
    logs) leet-logs ;;
    open) leet-open ;;
    *) echo "用法: $0 {start|stop|restart|status|rebuild|logs|open}" ;;
  esac
fi
