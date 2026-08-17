# 常驻服务(日常访问)与开发服务的分工

## 一句话

日常用**生产模式常驻**在 `localhost:3000`(`leet-start` 启停);开发调试用 `npm run dev` 跑在 `localhost:3001`。两者**构建目录、端口、进程全部分离**,可以同时开着互不干扰。

## 终端命令(已装进 ~/.zshrc)

| 命令 | 作用 |
|---|---|
| `leet-start` | 启动常驻服务(没有生产构建时自动先构建);已在运行则原样返回 |
| `leet-stop` | 停止;PID 文件丢失时会兜底清理端口上的残留进程 |
| `leet-restart` | 重启 |
| `leet-status` | 运行状态 · PID · 内存 · CPU · 访问地址 |
| `leet-rebuild` | **改过代码后**重新构建并重启 |
| `leet-logs` | 跟踪日志(`.leet/server.log`) |
| `leet-open` | 浏览器打开 |

换端口:`LEETPREP_PORT=3002 leet-start`。实现在 `scripts/leet-server.sh`,`~/.zshrc` 只是 source 它——改脚本立即生效,不用再动 zshrc。

## 什么时候需要重新构建

| 改动 | 是否需要 `leet-rebuild` |
|---|---|
| 加题、改题、写笔记、标高频、标已掌握 | **不需要**,刷新页面即见 |
| 新增/修改知识库文章、开源解读 | **不需要** |
| 改 `app/` `components/` `lib/` 等代码、装新依赖 | 需要 |

原因:所有页面都是 `export const dynamic = 'force-dynamic'`,每次请求实时读磁盘上的 markdown。

## 为什么 dev 和 prod 要分开构建目录

`npm run dev` 会持续改写构建目录,如果和生产共用 `.next`,常驻服务的产物会被冲掉,再次启动时报
`TypeError: routesManifest.dataRoutes is not iterable`(本项目踩过)。

现在 `next.config.ts` 用 `distDir: process.env.NEXT_DIST_DIR || '.next'`:

- 生产:`NEXT_DIST_DIR=.next-prod`(`npm run build` / `npm run start` 已内置)
- 开发:默认 `.next`

两个目录都在 `.gitignore` 里。

## 资源占用(本机实测)

| 状态 | 内存(RSS) | CPU |
|---|---|---|
| 空闲常驻 | 约 200 MB | ~0% |
| 处理请求 | 约 210 MB | 瞬时占用,请求完即回落 |

页面响应 5–40 ms。常年挂着的代价基本只是那 200 MB 内存;不想挂就 `leet-stop`,下次 `leet-start` 秒起(已有构建时不再重新构建)。

## 开机自启(可选,默认没做)

需要的话可以加一个 launchd 用户级 agent(`~/Library/LaunchAgents/`)在登录时自动 `leet-start`。当前是**手动启停**,符合"要用时再开"的习惯。
