#!/usr/bin/env bash
# 刷题系统常驻服务:隔离验证构建,以 blue/green 固定槽安全发布。

LEETPREP_DIR="${LEETPREP_DIR:-$HOME/1Project/interviewprep}"
LEETPREP_PORT="${LEETPREP_PORT:-3000}"

_leet_paths() {
  LEET_RUN_DIR="$LEETPREP_DIR/.leet"
  LEET_PID_FILE="$LEET_RUN_DIR/server.pid"
  LEET_SERVER_DIST_FILE="$LEET_RUN_DIR/server.dist"
  LEET_ACTIVE_DIST_FILE="$LEET_RUN_DIR/active-dist"
  LEET_MIGRATION_FILE="$LEET_RUN_DIR/blue-green-ready"
  LEET_LOCK_PATH="$LEET_RUN_DIR/operation.lock"
  LEET_SERVER_LOG="$LEET_RUN_DIR/server.log"
  LEET_BUILD_LOG="$LEET_RUN_DIR/build.log"
  mkdir -p "$LEET_RUN_DIR"
}

_leet_valid_dist() {
  case "${1:-}" in
    .next-blue|.next-green|.next-prod) return 0 ;;
    *) return 1 ;;
  esac
}

_leet_read_dist() {
  local file="$1" dist
  [ -f "$file" ] || return 1
  IFS= read -r dist <"$file" || return 1
  _leet_valid_dist "$dist" || return 1
  [ -f "$LEETPREP_DIR/$dist/BUILD_ID" ] || return 1
  printf '%s\n' "$dist"
}

_leet_atomic_write() {
  local file="$1" value="$2" tmp="$1.tmp.$$"
  printf '%s\n' "$value" >"$tmp" || return 1
  mv -f "$tmp" "$file"
}

_leet_commit_active() {
  local dist="$1"
  if [ "$dist" != '.next-prod' ]; then
    _leet_atomic_write "$LEET_MIGRATION_FILE" '1' || return 1
  fi
  _leet_atomic_write "$LEET_ACTIVE_DIST_FILE" "$dist"
}

_leet_inactive_dist() {
  case "${1:-}" in
    .next-blue) printf '%s\n' '.next-green' ;;
    .next-green|.next-prod|'') printf '%s\n' '.next-blue' ;;
    *) return 1 ;;
  esac
}

_leet_raw_pid() {
  [ -f "$LEET_PID_FILE" ] || return 1
  local pid
  pid=$(tr -dc '0-9' <"$LEET_PID_FILE" 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && printf '%s\n' "$pid"
}

_leet_pid_identity() {
  local pid="$1" cwd command
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)
  [ "$cwd" = "$LEETPREP_DIR" ] || return 1
  command=$(ps -p "$pid" -o command= 2>/dev/null) || return 1
  case "$command" in
    *node_modules/next/dist/bin/next*" start "*|*next-server*) return 0 ;;
    *) return 1 ;;
  esac
}

_leet_pid_owned() {
  local pid="$1"
  _leet_pid_identity "$pid" || return 1
  lsof -a -p "$pid" -iTCP:"$LEETPREP_PORT" -sTCP:LISTEN -t 2>/dev/null | grep -qx "$pid"
}

_leet_owned_pid() {
  local pid
  pid=$(_leet_raw_pid) || return 1
  _leet_pid_owned "$pid" || return 2
  printf '%s\n' "$pid"
}

_leet_actual_dist() {
  local pid="$1" dist
  _leet_pid_owned "$pid" || return 1
  if [ -e "$LEET_SERVER_DIST_FILE" ]; then
    _leet_read_dist "$LEET_SERVER_DIST_FILE"
  elif [ ! -e "$LEET_MIGRATION_FILE" ] && [ -f "$LEETPREP_DIR/.next-prod/BUILD_ID" ]; then
    printf '%s\n' '.next-prod'
  else
    return 1
  fi
}

_leet_port_owner() {
  lsof -ti:"$LEETPREP_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1
}

_leet_acquire_lock() {
  _leet_paths
  if [ ! -e "$LEET_LOCK_PATH" ] && [ ! -L "$LEET_LOCK_PATH" ] && ln -s "$$" "$LEET_LOCK_PATH" 2>/dev/null; then
    return 0
  fi
  local owner
  owner=$(readlink "$LEET_LOCK_PATH" 2>/dev/null)
  case "$owner" in ''|*[!0-9]*) echo "✗ 操作锁状态异常:$LEET_LOCK_PATH"; return 1 ;; esac
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    echo "✗ 另一个服务操作正在进行(PID $owner)"
    return 1
  fi
  rm -f "$LEET_LOCK_PATH" 2>/dev/null || return 1
  ln -s "$$" "$LEET_LOCK_PATH" 2>/dev/null || {
    echo "✗ 另一个服务操作抢先获得了锁"
    return 1
  }
}

_leet_release_lock() {
  [ "$(readlink "$LEET_LOCK_PATH" 2>/dev/null)" = "$$" ] && rm -f "$LEET_LOCK_PATH" 2>/dev/null
}

_leet_healthcheck() {
  local html deadline asset css_count=0 js_count=0 assets ok
  html=$(mktemp "$LEET_RUN_DIR/health.XXXXXX") || return 1
  deadline=$(($(date +%s) + 7))
  while [ "$(date +%s)" -le "$deadline" ]; do
    if curl -fsS --connect-timeout 1 --max-time 1 -o "$html" "http://localhost:$LEETPREP_PORT/" 2>/dev/null; then
      assets=$(grep -Eo '/_next/static/[^" ]+\.(css|js)' "$html" | sort -u)
      if [ -n "$assets" ]; then
        ok=1
        while IFS= read -r asset; do
          [ -n "$asset" ] || continue
          case "$asset" in *.css) css_count=$((css_count + 1)) ;; *.js) js_count=$((js_count + 1)) ;; esac
          if ! curl -fsS --connect-timeout 1 --max-time 1 -o /dev/null "http://localhost:$LEETPREP_PORT$asset" 2>/dev/null; then
            ok=0
            break
          fi
          [ "$(date +%s)" -le "$deadline" ] || { ok=0; break; }
        done <<<"$assets"
        if [ "$ok" -eq 1 ] && [ "$css_count" -gt 0 ] && [ "$js_count" -gt 0 ]; then
          rm -f "$html"
          return 0
        fi
      fi
    fi
    css_count=0
    js_count=0
    sleep 0.25
  done
  rm -f "$html"
  return 1
}

_leet_build_dist() {
  local dist="$1"
  case "$dist" in .next-blue|.next-green) ;; *) return 1 ;; esac
  cd "$LEETPREP_DIR" || return 1
  echo "· 构建候选槽 $dist(在线服务保持运行)…"
  NEXT_DIST_DIR="$dist" npm run build:release >"$LEET_BUILD_LOG" 2>&1 || {
    echo "✗ 构建失败,看日志:$LEET_BUILD_LOG"
    return 1
  }
  [ -f "$LEETPREP_DIR/$dist/BUILD_ID" ] || {
    echo "✗ 构建未生成 BUILD_ID:$dist"
    return 1
  }
  echo "✓ 候选构建完成:$dist"
}

_leet_start_dist() {
  local dist="$1" owner pid actual
  _leet_valid_dist "$dist" && [ -f "$LEETPREP_DIR/$dist/BUILD_ID" ] || {
    echo "✗ 不能启动无效构建槽:$dist"
    return 1
  }
  owner=$(_leet_port_owner)
  if [ -n "$owner" ]; then
    echo "✗ 端口 $LEETPREP_PORT 已被 PID $owner 占用;不会自动终止未知进程"
    return 1
  fi
  : >"$LEET_SERVER_LOG"
  LEET_NODE_BIN="$(command -v node)" python3 "$LEETPREP_DIR/scripts/leet-daemon.py" \
    "$LEETPREP_DIR" "$LEETPREP_PORT" "$LEET_SERVER_LOG" "$LEET_PID_FILE" \
    "$dist" "$LEET_SERVER_DIST_FILE" || return 1
  if _leet_healthcheck; then
    pid=$(_leet_owned_pid) || { echo "✗ 页面健康但无法确认目标进程"; return 1; }
    actual=$(_leet_read_dist "$LEET_SERVER_DIST_FILE") || { echo "✗ 页面健康但实际槽状态无效"; return 1; }
    [ "$actual" = "$dist" ] || { echo "✗ 页面健康但实际槽为 $actual,不是 $dist"; return 1; }
    echo "✓ 服务健康:$dist · PID $(_leet_raw_pid)"
    return 0
  fi
  echo "✗ 服务健康检查失败:$dist"
  return 1
}

_leet_terminate_pid() {
  local pid="$1" i
  kill "$pid" 2>/dev/null || return 1
  for i in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null; then
    _leet_pid_identity "$pid" || { echo "✗ PID 状态在停止时改变;拒绝强制终止"; return 1; }
    kill -9 "$pid" 2>/dev/null || return 1
  fi
  for i in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null && [ -z "$(_leet_port_owner)" ]; then break; fi
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null || [ -n "$(_leet_port_owner)" ]; then
    echo "✗ 进程或端口未在超时内释放"
    return 1
  fi
  rm -f "$LEET_PID_FILE" "$LEET_SERVER_DIST_FILE"
}

_leet_stop_candidate() {
  local expected="$1" pid actual owner
  pid=$(_leet_raw_pid) || {
    owner=$(_leet_port_owner)
    [ -z "$owner" ] || { echo "✗ 候选 PID 丢失且端口被 PID $owner 占用"; return 1; }
    rm -f "$LEET_PID_FILE" "$LEET_SERVER_DIST_FILE"
    return 0
  }
  _leet_pid_identity "$pid" || { echo "✗ 候选 PID 归属异常;拒绝终止"; return 1; }
  actual=$(_leet_read_dist "$LEET_SERVER_DIST_FILE") || { echo "✗ 候选槽状态无效;拒绝终止"; return 1; }
  [ "$actual" = "$expected" ] || { echo "✗ 候选实际槽 $actual 与预期 $expected 不一致;拒绝终止"; return 1; }
  _leet_terminate_pid "$pid"
}

_leet_stop_impl() {
  local pid rc owner
  pid=$(_leet_owned_pid)
  rc=$?
  if [ "$rc" -eq 2 ]; then
    echo "✗ PID 文件指向的进程不属于本项目;拒绝停止"
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    rm -f "$LEET_PID_FILE" "$LEET_SERVER_DIST_FILE"
    owner=$(_leet_port_owner)
    if [ -n "$owner" ]; then
      echo "✗ 端口 $LEETPREP_PORT 被未知 PID $owner 占用;拒绝停止"
      return 1
    fi
    echo "· 服务本来就没在运行"
    return 0
  fi
  _leet_terminate_pid "$pid" || return 1
  echo "✓ 已停止(PID $pid)"
}

_leet_start_impl() {
  local pid rc actual active dist
  pid=$(_leet_owned_pid)
  rc=$?
  if [ "$rc" -eq 0 ]; then
    actual=$(_leet_actual_dist "$pid") || { echo "✗ 无法确认当前进程使用的构建槽"; return 1; }
    if active=$(_leet_read_dist "$LEET_ACTIVE_DIST_FILE"); then
      [ "$active" = "$actual" ] || { echo "✗ 状态错代:实际 $actual,健康槽 $active;请先 leet-status"; return 1; }
    fi
    _leet_healthcheck || { echo "✗ 进程存在但页面资源不健康;运行 leet-restart 重新加载同一构建"; return 1; }
    _leet_commit_active "$actual" || { echo "✗ 无法确认健康槽状态;后续操作将 fail closed"; return 1; }
    echo "✓ 已在运行(PID $pid) · 槽 $actual → http://localhost:$LEETPREP_PORT"
    return 0
  elif [ "$rc" -eq 2 ]; then
    echo "✗ PID 文件指向的进程不属于本项目;拒绝启动"
    return 1
  fi

  if [ -e "$LEET_ACTIVE_DIST_FILE" ]; then
    active=$(_leet_read_dist "$LEET_ACTIVE_DIST_FILE") || { echo "✗ active-dist 无效;拒绝启动"; return 1; }
    dist="$active"
  elif [ ! -e "$LEET_MIGRATION_FILE" ] && [ -f "$LEETPREP_DIR/.next-prod/BUILD_ID" ]; then
    dist='.next-prod'
  elif [ -e "$LEET_MIGRATION_FILE" ]; then
    echo "✗ 已完成 blue/green 迁移但 active-dist 缺失;拒绝猜测"
    return 1
  else
    dist='.next-blue'
    _leet_build_dist "$dist" || return 1
  fi
  _leet_start_dist "$dist" || { _leet_stop_candidate "$dist" >/dev/null 2>&1 || true; return 1; }
  _leet_commit_active "$dist" || { echo "✗ 无法提交健康槽状态;停止未提交服务"; _leet_stop_candidate "$dist" >/dev/null 2>&1 || true; return 1; }
  echo "  http://localhost:$LEETPREP_PORT · 关闭终端不影响服务"
}

_leet_restart_impl() {
  local pid rc actual active dist
  pid=$(_leet_owned_pid)
  rc=$?
  if [ "$rc" -eq 0 ]; then
    actual=$(_leet_actual_dist "$pid") || { echo "✗ 无法确认当前进程使用的构建槽"; return 1; }
    if active=$(_leet_read_dist "$LEET_ACTIVE_DIST_FILE"); then
      [ "$active" = "$actual" ] || { echo "✗ 状态错代:实际 $actual,健康槽 $active;拒绝猜测"; return 1; }
    fi
    dist="$actual"
    _leet_stop_impl || return 1
  elif [ "$rc" -eq 2 ]; then
    echo "✗ PID 文件指向的进程不属于本项目;拒绝重启"
    return 1
  elif [ -e "$LEET_ACTIVE_DIST_FILE" ]; then
    dist=$(_leet_read_dist "$LEET_ACTIVE_DIST_FILE") || { echo "✗ active-dist 无效;拒绝重启"; return 1; }
  elif [ ! -e "$LEET_MIGRATION_FILE" ] && [ -f "$LEETPREP_DIR/.next-prod/BUILD_ID" ]; then
    dist='.next-prod'
  else
    echo "✗ 没有可重启的构建;请运行 leet-rebuild"
    return 1
  fi
  _leet_start_dist "$dist" || { _leet_stop_candidate "$dist" >/dev/null 2>&1 || true; return 1; }
  _leet_commit_active "$dist" || { echo "✗ 无法提交健康槽状态;停止未提交服务"; _leet_stop_candidate "$dist" >/dev/null 2>&1 || true; return 1; }
}

_leet_rebuild_impl() {
  local pid rc previous active candidate was_running=0
  pid=$(_leet_owned_pid)
  rc=$?
  if [ "$rc" -eq 0 ]; then
    was_running=1
    previous=$(_leet_actual_dist "$pid") || { echo "✗ 无法确认当前进程使用的构建槽"; return 1; }
    if active=$(_leet_read_dist "$LEET_ACTIVE_DIST_FILE"); then
      [ "$active" = "$previous" ] || { echo "✗ 状态错代:实际 $previous,健康槽 $active;拒绝发布"; return 1; }
    fi
  elif [ "$rc" -eq 2 ]; then
    echo "✗ PID 文件指向的进程不属于本项目;拒绝发布"
    return 1
  elif [ -e "$LEET_ACTIVE_DIST_FILE" ]; then
    previous=$(_leet_read_dist "$LEET_ACTIVE_DIST_FILE") || { echo "✗ active-dist 无效;拒绝发布"; return 1; }
  elif [ ! -e "$LEET_MIGRATION_FILE" ] && [ -f "$LEETPREP_DIR/.next-prod/BUILD_ID" ]; then
    previous='.next-prod'
  elif [ -e "$LEET_MIGRATION_FILE" ]; then
    echo "✗ 已完成 blue/green 迁移但 active-dist 缺失;拒绝发布"
    return 1
  else
    previous=''
  fi

  candidate=$(_leet_inactive_dist "$previous") || return 1
  _leet_build_dist "$candidate" || return 1
  if [ "$was_running" -eq 1 ]; then _leet_stop_impl || return 1; fi
  if _leet_start_dist "$candidate"; then
    if _leet_commit_active "$candidate"; then
      echo "✓ 发布完成:$candidate → http://localhost:$LEETPREP_PORT"
      return 0
    fi
    echo "✗ 候选服务健康但状态提交失败;开始回滚"
  fi

  if ! _leet_stop_candidate "$candidate" >/dev/null 2>&1; then
    echo "✗ 无法停止候选进程;未尝试启动第二个服务,请运行 leet-status"
    return 1
  fi
  if [ -n "$previous" ] && _leet_start_dist "$previous"; then
    if _leet_commit_active "$previous"; then
      echo "✗ 候选槽失败,已回滚到 $previous"
    else
      echo "✗ 旧槽已恢复健康,但 active-dist 提交失败;后续操作将 fail closed"
    fi
  else
    if [ -n "$(_leet_port_owner)" ]; then
      echo "✗ 候选槽失败且回滚失败;端口仍有进程,请运行 leet-status"
    else
      echo "✗ 候选槽失败且回滚失败;服务当前已停止"
    fi
  fi
  return 1
}

_leet_status_impl() {
  _leet_paths
  local pid rc actual active build_id mem cpu tty
  active=$(_leet_read_dist "$LEET_ACTIVE_DIST_FILE" 2>/dev/null || true)
  pid=$(_leet_owned_pid)
  rc=$?
  if [ "$rc" -eq 0 ]; then
    actual=$(_leet_actual_dist "$pid" 2>/dev/null || true)
    build_id=$([ -n "$actual" ] && sed -n '1p' "$LEETPREP_DIR/$actual/BUILD_ID" 2>/dev/null)
    mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f MB", $1/1024}')
    cpu=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')
    tty=$(ps -o tty= -p "$pid" 2>/dev/null | tr -d ' ')
    echo "✓ 运行中 · PID $pid · 实际槽 ${actual:-未知} · 健康槽 ${active:-未记录}"
    echo "  BUILD_ID ${build_id:-未知} · 内存 ${mem:-?} · CPU ${cpu:-?}% · TTY ${tty:-??}"
    [ -z "$active" ] || [ "$active" = "$actual" ] || { echo "✗ 状态错代,服务操作将 fail closed"; return 1; }
  elif [ "$rc" -eq 2 ]; then
    echo "✗ PID 文件归属异常;健康槽 ${active:-未记录}"
    return 1
  else
    echo "· 未运行 · 健康槽 ${active:-未记录}(leet-start 启动)"
  fi
}

_leet_run_locked() {
  _leet_acquire_lock || return 1
  trap '_leet_release_lock' EXIT INT TERM HUP
  "$@"
  local rc=$?
  _leet_release_lock
  trap - EXIT INT TERM HUP
  return "$rc"
}

# source 后保留熟悉的全局命令,实际操作进入独立脚本进程以可靠持锁。
leet-start() { LEETPREP_DIR="$LEETPREP_DIR" LEETPREP_PORT="$LEETPREP_PORT" "$LEETPREP_DIR/scripts/leet-server.sh" start; }
leet-stop() { LEETPREP_DIR="$LEETPREP_DIR" LEETPREP_PORT="$LEETPREP_PORT" "$LEETPREP_DIR/scripts/leet-server.sh" stop; }
leet-restart() { LEETPREP_DIR="$LEETPREP_DIR" LEETPREP_PORT="$LEETPREP_PORT" "$LEETPREP_DIR/scripts/leet-server.sh" restart; }
leet-rebuild() { LEETPREP_DIR="$LEETPREP_DIR" LEETPREP_PORT="$LEETPREP_PORT" "$LEETPREP_DIR/scripts/leet-server.sh" rebuild; }
leet-status() { LEETPREP_DIR="$LEETPREP_DIR" LEETPREP_PORT="$LEETPREP_PORT" "$LEETPREP_DIR/scripts/leet-server.sh" status; }
leet-logs() { _leet_paths; tail -f "$LEET_SERVER_LOG"; }
leet-build-logs() { _leet_paths; tail -f "$LEET_BUILD_LOG"; }
leet-open() { open "http://localhost:$LEETPREP_PORT"; }

_LEET_SCRIPT_SOURCED=0
if [ -n "${ZSH_EVAL_CONTEXT:-}" ]; then
  case "$ZSH_EVAL_CONTEXT" in *:file) _LEET_SCRIPT_SOURCED=1 ;; esac
elif [ "${BASH_SOURCE[0]}" != "$0" ]; then
  _LEET_SCRIPT_SOURCED=1
fi

if [ "$_LEET_SCRIPT_SOURCED" -eq 0 ] && [ -n "${1:-}" ]; then
  _leet_paths
  case "$1" in
    start) _leet_run_locked _leet_start_impl ;;
    stop) _leet_run_locked _leet_stop_impl ;;
    restart) _leet_run_locked _leet_restart_impl ;;
    rebuild) _leet_run_locked _leet_rebuild_impl ;;
    status) _leet_status_impl ;;
    logs) tail -f "$LEET_SERVER_LOG" ;;
    build-logs) tail -f "$LEET_BUILD_LOG" ;;
    open) open "http://localhost:$LEETPREP_PORT" ;;
    *) echo "用法: $0 {start|stop|restart|rebuild|status|logs|build-logs|open}"; exit 2 ;;
  esac
fi
