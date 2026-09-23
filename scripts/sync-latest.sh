#!/usr/bin/env bash
# 本机会话启动时自动同步云端每日解读(docs/10-材料解读流程.md 第二节「来源二」末段)
#
# 由 Claude Code 与 Codex 的 SessionStart 钩子调用,也可以手动跑。输出是给 agent 的简报。
# 只做快进合并:本机有未推送的提交、或未提交的改动与云端改了同一文件时,git 自己会拒绝,
# 这里只报告、不 rebase、不 stash、不动任何东西。几个会话同时启动时只有一个在同步。
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 0
say() { echo "[云端同步] $*"; }

LOCK="$REPO/.git/sync-latest.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  # 锁超过 10 分钟视为上次异常退出留下的
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +10 2>/dev/null)" ]; then
    rmdir "$LOCK" 2>/dev/null && mkdir "$LOCK" 2>/dev/null || exit 0
  else
    exit 0
  fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

branch="$(git symbolic-ref --short -q HEAD)"
if [ "$branch" != "main" ]; then
  say "当前在 ${branch:-游离 HEAD},不是 main,跳过"
  exit 0
fi

# macOS 没有 timeout;用 perl 的 alarm 给 fetch 限时,断网时不拖住会话启动
if ! perl -e 'alarm 15; exec @ARGV' git fetch -q origin main 2>/dev/null; then
  say "连不上 GitHub(15 秒超时),本次跳过"
  exit 0
fi

old="$(git rev-parse HEAD)"
remote="$(git rev-parse origin/main)"
if [ "$old" = "$remote" ]; then
  say "已与 GitHub 一致"
  exit 0
fi
if git merge-base --is-ancestor origin/main HEAD; then
  say "本机有 $(git rev-list --count origin/main..HEAD) 个提交还没推送,GitHub 上没有新东西"
  exit 0
fi
if ! git merge-base --is-ancestor HEAD origin/main; then
  say "本机与 GitHub 已分叉(本机领先 $(git rev-list --count origin/main..HEAD)、落后 $(git rev-list --count HEAD..origin/main)),不自动合并。先把本机提交推上去或 rebase 后再同步"
  exit 0
fi

if ! out="$(git merge --ff-only -q origin/main 2>&1)"; then
  say "GitHub 上有 $(git rev-list --count HEAD..origin/main) 个新提交,但快进合并被拒绝:本机未提交的改动与它们改了同一批文件。你的改动原样保留,处理完再同步。git 的原话:"
  echo "$out" | sed -n '1,15p'
  exit 0
fi

new="$(git rev-parse HEAD)"
say "已快进 $(git rev-list --count "$old..$new") 个提交(${old:0:7} → ${new:0:7})"

published="$(git -c core.quotepath=off diff --name-only --diff-filter=A "$old" "$new" -- reports readings \
  | grep -E '\.md$' | grep -vE '/_|/index\.md$' || true)"
if [ -n "$published" ]; then
  echo "新发布的解读:"
  echo "$published" | sed 's/^/  - /'
fi

daily="$(git log --format='%H' --grep='^docs(daily)' "$old..$new" | head -1)"
if [ -n "$daily" ]; then
  echo "云端最近一次日报的提交说明:"
  git log -1 --format='%B' "$daily" | grep -v '^Co-Authored-By' | sed 's/^/  /'
fi

# 补原件与重启服务都放后台,不拖慢会话启动;结果写进 .git 下的日志
if git diff --quiet "$old" "$new" -- 'readings/_alphaxiv-扫描记录.md'; then :; else
  nohup node scripts/papers-pull.mjs --apply >"$REPO/.git/sync-latest-pull.log" 2>&1 &
  say "后台补下载新原件,日志 .git/sync-latest-pull.log"
fi
if [ -n "$(git diff --name-only "$old" "$new" -- public | head -1)" ]; then
  nohup "$REPO/scripts/leet-server.sh" restart >"$REPO/.git/sync-latest-restart.log" 2>&1 &
  say "有新配图,后台 leet-restart,日志 .git/sync-latest-restart.log"
fi
exit 0
