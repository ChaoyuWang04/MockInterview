# 装一个 skill,让两边都能用

**现行做法**:skill 放 `.claude/skills/<名字>/SKILL.md`,`.agents/skills` 是指向 `.claude/skills` 的软链接。外部 skill 先完整读、不执行其中脚本,按本仓库手册改写后再放进来;规则与流程留在手册,skill 只装需要时才加载的操作细节。

**原则**:一份真源,两边读;按需加载的东西才适合做成 skill。

**Claude 与 Codex**:Claude Code 读 `.claude/skills/`;Codex 从当前目录到仓库根逐级扫 `.agents/skills/`,用户级在 `~/.agents/skills/`。

**本仓库落实在**:`.claude/skills/`、`.agents/skills`(软链接)

**来源**:<https://learn.chatgpt.com/docs/build-skills>(原 developers.openai.com/codex/skills) · 核实于 2026-09-24
