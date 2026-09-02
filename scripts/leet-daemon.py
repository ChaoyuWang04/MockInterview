#!/usr/bin/env python3
"""把 next start 变成真正脱离终端的后台进程。

用法(由 scripts/leet-server.sh 调用):
    python3 leet-daemon.py <APP_DIR> <PORT> <LOG_FILE> <PID_FILE>

为什么需要它:`nohup cmd &` 起的进程仍留在终端的会话里,关闭 Terminal 窗口
时会被一起收走。这里用标准的 double-fork + setsid 把进程放进**新的会话**,
父进程变成 init(PPID=1)、无控制终端(TTY=??),终端怎么关都影响不到它。

历史备注:项目原先放在 ~/Desktop(macOS 隐私保护 TCC 目录),launchd 拉起的
进程读不到里面的文件(实测报 `can't open input file`),所以只能用这套方案。
2026-09-02 已迁到 ~/1Project,TCC 限制不再存在;这套 double-fork 方案照常可用,
若要改用 LaunchAgent(顺带获得开机自启)现在也没有障碍了。
"""

import os
import sys


def main() -> None:
    if len(sys.argv) != 5:
        sys.exit(f"用法: {sys.argv[0]} <APP_DIR> <PORT> <LOG_FILE> <PID_FILE>")
    app_dir, port, log_file, pid_file = sys.argv[1:5]

    if os.fork() > 0:  # 父进程立即返回,不阻塞调用它的 shell
        return
    os.setsid()  # 脱离原会话,成为新会话的首进程(不再关联任何终端)
    if os.fork() > 0:  # 二次 fork:非会话首进程,永远不会再获得控制终端
        os._exit(0)

    # 注:不靠忽略 SIGHUP 来保命 —— Node 启动时会重置该信号的处理方式,
    # 连 `nohup node` 都挡不住直接发来的 SIGHUP(已实测)。
    # 真正的保障是上面的 setsid:进程没有控制终端,关闭终端时压根不会收到 SIGHUP。
    os.chdir(app_dir)
    log = open(log_file, "ab", buffering=0)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)

    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    env = dict(os.environ, NEXT_DIST_DIR=".next-prod", NODE_ENV="production")
    node = env.get("LEET_NODE_BIN") or "node"
    next_bin = os.path.join(app_dir, "node_modules", "next", "dist", "bin", "next")
    os.execvpe(node, [node, next_bin, "start", "-p", str(port)], env)


if __name__ == "__main__":
    main()
