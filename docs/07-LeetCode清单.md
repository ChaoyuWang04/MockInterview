# LeetCode 热题 100 模块

定位:一份**只读清单 + 极轻量便签**。不做刷题状态、不做题解沉淀(那是题库和知识库的事),只回答两个问题——这 100 题长什么样、我当时记的小技巧是什么。

## 文件与路由

| 内容 | 位置 |
|---|---|
| 题目清单(数据源) | `leetcode/hot100.md` —— `## 分组` + 四列表格 `题号 \| 标题 \| slug \| 难度` |
| 补充题 | `leetcode/extra.md` —— 真题中出现但不在官方 100 内的题;格式与 hot100.md 相同 |
| 高频标记 | `leetcode/high-freq.md` —— 每行 `- <slug>`;**与清单分开存放**,重新同步清单不丢标记 |
| 笔记 | `leetcode/notes/<slug>.md` —— 一题一文件,按需创建 |
| 页面 | `/leetcode`(主页入口「💯 LeetCode 热题」) |
| 写接口 | `PATCH /api/leetcode-note` `{ slug, note }`、`PATCH /api/leetcode-hot` `{ slug, hot }`,slug 均须在清单内 |

`slug` 是主键,leetcode.com 与 leetcode.cn 共用同一个 slug,两个跳转按钮各自拼 `https://leetcode.com/problems/<slug>/` 与 `https://leetcode.cn/problems/<slug>/`。

## 补充题怎么加

真题里遇到 hot100 之外的题:加一行到 `leetcode/extra.md`。**不要写进 hot100.md**——那个文件会被 `npm run hot100:sync` 整体覆盖。

- 分组名与 hot100 相同 → 自动并入该组(如「链表」);
- 分组名不存在 → 作为新分组追加在列表末尾(如「排序与字符串」);
- 页面上补充题带灰色「补」标记,其余行为(笔记、高频、排序)与官方题完全一致。

### 混合输入怎么分流

一次输入同时包含 LeetCode 标准题、题目变体和大模型手撕题时,先逐题确认真实题干,再按下面的顺序分流:

1. 用题意核对 LeetCode 官方题号与 slug,不能因为老师答案碰巧选择了某道题,就把「任选一道算法题」之类的开放题硬映射成该题。
2. 在 `hot100.md` 与 `extra.md` 中同时按题号和 slug 去重。官方 100 已有的题不重复记录;两个文件都没有的标准题才写入 `extra.md`。
3. 页面上的灰色「补」由 `extra.md` 自动产生,不需要另存标签;除非用户明确要求,不要同时写入 `high-freq.md`。
4. 非标准题、复合题或大模型实现题进入 `questions/手撕代码/`,按题库规则合并近义题并保存答案。与某道 LeetCode 相似只能作为关联,不能代替这道独立真题。
5. LeetCode 清单只保存题号、标题、slug 与难度。老师答案和逐图纠错直接用于核对对应真题或清单,不另存副本,也不批量创建 `leetcode/notes/` 题解。

## 页面行为

- 按官方 17 个专题分组;每行:题号 · 标题(点击去**力扣中国 .cn**)· **「US」按钮(去 leetcode.com)** · 📝 笔记按钮 · 难度(简单绿/中等黄/困难红)· 「高」高频开关
- 两个站点共用同一个 slug,所以两个入口只是域名不同;笔记面板底部也保留了这两个跳转
- **组内排序规则**(`lib/leetcode-sort.ts`,纯函数、有单测):高频题整体排在前面,**两个分桶内部都按 简单 → 中等 → 困难**;全是高频或全不是高频时自然退化为纯难度排序;同桶同难度保持清单原始顺序(稳定排序)
- 点「高」即时重排并写回 `high-freq.md`;写回失败会回滚页面状态,保证页面与文件一致
- 📝 打开右下角悬浮小窗:标题 + 关闭(× 或 Esc)、**纯 textarea 原始文本不渲染**、底部两个跳转按钮(LeetCode US / 力扣中国)与保存状态
- 笔记**输入停顿 800ms 自动保存**,关闭时兜底保存一次;有笔记的题目 📝 按钮显示为蓝色
- 清空内容保存 = 删除该笔记文件(不留空文件)

## 清单校准(可选,推荐做一次)

清单已按用户提供的官方页面截图逐组核对(2026-08-14:17 组 100 题全对)。想用官方数据一键覆盖(本机开发环境抓不到外网时在你自己的终端跑):

```bash
curl -s -X POST https://leetcode.cn/graphql/ -H 'Content-Type: application/json' -H 'User-Agent: Mozilla/5.0' -d '{"query":"query q($slug: String!){ studyPlanV2Detail(planSlug:$slug){ planSubGroups { name questions { frontendQuestionId title translatedTitle titleSlug difficulty } } } }","variables":{"slug":"top-100-liked"}}' -o /tmp/hot100.json
```

然后覆盖并校验:

```bash
npm run hot100:sync /tmp/hot100.json && npm test
```

`tests/leetcode.test.ts` 会守住结构:**恰好 17 组 100 题**、题号为数字、slug 形如 `a-b-c`、难度三选一、slug/题号不重复。校准后若数量变化(官方调整过题单),按实际数字更新测试里的断言。

## 维护约定

- 清单是参考数据,平时不手改;要改就走上面的同步脚本,避免手滑破坏表格结构
- 笔记文件很小、纯文本,**整文件覆盖写回**(不同于题目 Note 的定点替换——笔记文件里没有别的内容需要保护)
- 验证一律 `npm test`,不开浏览器
