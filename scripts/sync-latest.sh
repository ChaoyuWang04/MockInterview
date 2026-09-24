#!/usr/bin/env bash
# 本机会话启动时自动同步云端每日解读(docs/10-材料解读流程.md 第二节「来源二」末段)
#
# 由 Claude Code 与 Codex 的 SessionStart 钩子、Hermes 的 pre_llm_call 钩子(只在会话第一轮)调用,也可以手动跑。输出是给 agent 的简报。
# 只做快进合并:本机有未推送的提交、或未提交的改动与云端改了同一文件时,git 自己会拒绝,
# 这里只报告、不 rebase、不 stash、不动任何东西。几个会话同时启动时只有一个在同步。
# 开头另报内容目录里没提交的文件,提醒「一批一推」(AGENTS.md 协作规则)。
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

# 未收尾提醒:内容目录里改了没提交的文件。只报告,不替任何一条线提交(AGENTS.md「一批一推」)
CONTENT_DIRS="reports readings opensource knowledge questions leetcode meetings contrib public"
# -z 输出:路径不加引号,空格与中文原样;重命名条目后面多跟一个原路径,跳过它
# shellcheck disable=SC2086
pending="$(git status --porcelain -z -uall -- $CONTENT_DIRS 2>/dev/null | perl -0ne '
  chomp; if ($skip) { $skip = 0; next }
  my ($st, $f) = (substr($_, 0, 2), substr($_, 3)); $skip = 1 if $st =~ /[RC]/;
  $n++; $d{(split m{/}, $f)[0]}++;
  my @s = stat $f; $m = $s[9] if @s && (!defined $m || $s[9] < $m);
  END { exit unless $n; print "$n\t", join("、", map { "$_ $d{$_}" } sort keys %d), "\t", (defined $m ? int((time - $m) / 86400) : "") }')"
if [ -n "$pending" ]; then
  IFS="$(printf '\t')" read -r n per_dir oldest <<EOF_P
$pending
EOF_P
  say "未收尾:内容目录有 $n 个文件改了没提交($per_dir)${oldest:+,最早的改于 $oldest 天前}。属于本线的先整批提交并推送;是别的线的就留给那条线"
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
  NODE_USE_ENV_PROXY=1 nohup node scripts/papers-pull.mjs --apply >"$REPO/.git/sync-latest-pull.log" 2>&1 &
  say "后台补下载新原件,日志 .git/sync-latest-pull.log"
fi
if [ -n "$(git diff --name-only "$old" "$new" -- public | head -1)" ]; then
  nohup "$REPO/scripts/leet-server.sh" restart >"$REPO/.git/sync-latest-restart.log" 2>&1 &
  say "有新配图,后台 leet-restart,日志 .git/sync-latest-restart.log"
fi
exit 0
