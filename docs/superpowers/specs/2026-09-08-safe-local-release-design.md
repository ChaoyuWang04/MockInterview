# 本地常驻服务安全发布设计

## 目标

消除运行中的 Next.js 进程与被原地覆盖的构建目录错代,使普通 `npm run build` 永不影响 `localhost:3000`,并让生产重建自动完成候选构建、短暂停机切换、CSS 健康检查和失败回滚。

## 已确认根因

当前 `npm run build` 与常驻服务共用 `.next-prod`:旧进程缓存旧 chunk 清单,新构建删除旧哈希资源后,旧 HTML 继续引用不存在的 CSS/JS。浏览器因而得到完整服务端 HTML,但退化成默认样式。

## 方案

采用固定双槽发布:

- `.next-check`:普通 `npm run build` 的隔离验证目录。
- `.next-blue` / `.next-green`:两个生产发布槽。运行 blue 时只构建 green,运行 green 时只构建 blue。
- `.leet/active-dist`:最近一次通过健康检查的发布槽。
- `.leet/server.dist`:当前进程实际使用的发布槽。
- `.next-prod`:只作一次性兼容旧安装;首次安全重建后不再写入。

`leet-rebuild` 根据**当前进程实际槽**选择未运行槽并完成构建,不能只相信 active-dist。成功后停止旧进程、启动新进程,请求首页并逐一验证同源 `/_next/static/` CSS 与关键 JS;只有健康检查通过才原子更新 active-dist。启动或健康检查失败时先停止候选进程,再启动旧槽并重新健康检查;回滚也失败时明确报告服务已停止。构建失败发生在切换前,旧服务不受影响。

`leet-restart` 不构建、不换槽,只让当前确认的构建重新加载;用于 `public/` 新文件、进程恢复和一次性修复错代。`leet-start` 只启动已确认健康的 active-dist;已有进程的状态矛盾时拒绝猜测并报告。迁移期若项目 PID 确属本服务但没有 server.dist,将实际槽视为 `.next-prod`。`leet-status` 显示 PID、实际槽、健康槽和 BUILD_ID,并报告错代或未知状态。

## 工作流

- Markdown 内容:刷新即可,不构建、不重启。
- 报告/代码验证:`npm run build` 写 `.next-check`,不部署。
- `public/` 静态资源变化:`leet-restart`。
- `app/`、`components/`、`lib/`、配置或依赖变化:`leet-rebuild`。
- 自动化只发生在明确的服务命令边界,不使用后台文件监听器,避免模拟面试进行中被任意重启。

## 安全边界

- 发布槽只接受 `.next-blue`、`.next-green` 和迁移期 `.next-prod`,拒绝任意路径。
- 新槽构建成功前不停止旧服务。
- active-dist 始终表示最近一次通过健康检查的槽;server.dist 始终表示当前项目进程实际使用的槽。状态文件以同目录临时文件加原子 rename 写入。
- 所有启停和发布操作用项目内原子锁串行化;锁记录操作进程 PID,只在该 PID 已不存在时回收陈旧锁。
- PID 必须同时匹配本项目目录、Next 启动命令和目标端口。未知端口占用或 PID 归属不明时只拒绝并报告,绝不按端口盲杀。
- 健康检查验证首页 200、至少一个 CSS、全部同源 `/_next/static/` CSS 与关键 JS 均成功;每次请求设置连接和总超时并有限重试。正常切换从停旧到新服务健康控制在 10 秒内;失败后的回滚可能更久。
- 回滚保留上一槽;两个固定槽避免无限增长和动态 distDir 污染 `tsconfig.json`。
- 构建日志与运行日志分离,避免构建时截断在线服务日志。

## 启动与迁移状态

- 全新环境:构建 blue,启动并检查健康,再写 active-dist。
- 只有 legacy 构建且服务未运行:`leet-start` 可启动 `.next-prod`;`leet-rebuild` 随后迁到 blue。
- legacy 服务正在运行且缺 server.dist:PID 归属验证通过后按 `.next-prod` 处理,因此当前“旧进程 + 被覆盖目录”可由 `leet-restart` 重新加载,再由 `leet-rebuild` 迁移。
- blue/green 正常运行:只构建当前实际槽的另一槽。
- active-dist/server.dist 非法、BUILD_ID 缺失或两者出现无法解释的矛盾:fail closed,不覆盖槽、不停止进程。
- `npm run build` 只能写 `.next-check`;`npm start` 走受控服务脚本,生产构建只能由发布脚本向白名单槽传入 distDir。

## 验收

自动测试守住验证目录与生产槽分离、实际槽优先选择、路径白名单、锁、PID 归属、daemon 使用指定槽、原子状态和文档口径一致。真实验收覆盖:修复当前旧进程、执行安全重建、在新服务运行时再跑 `npm run build`,确认首页及其 CSS/JS 始终为 200;另验证构建失败不改变在线 BUILD_ID 与资源响应。
