# pytorch 贡献台账

> 上游:<https://github.com/pytorch/pytorch> · 解读:`opensource/框架内核/torch-compile/`、`opensource/分布式训练/FSDP/`。规则见 [14-开源贡献](../docs/14-开源贡献.md)。
>
> 巨型仓(历史打包 2.2 GiB),14 的实验层流程暂不适用;目前只走「先提 issue」,真要提 PR 时再定接入方式。PyTorch 要求新贡献者的 PR 必须对应一个被标为 actionable 的 issue,AI 协助的内容要标明并由本人负责。

| 选题 | 上游 issue | 状态 | PR | 结论 |
|---|---|---|---|---|
| torch.compile 文档四处与代码不符:日志名 `aot_joint_graph` 写成复数、`unique_kernel_names` 与 `graph_partition` 的默认值写反、一处配置路径少了 `triton.` | 无 | 候选 | | 上游要不要:文档勘误,照文档设日志名会直接报错;有没有人在做:2026-09-24 搜过 issue 与 PR,无;怎么验证:逐处对到上游最新源码的行号 |
| DDP 与 torch.compile 谁先包:DDP 说明页说先包 DDP 再编译,编程模型页与排障页说先编译里面的模块 | 无 | 候选 | | 上游要不要:两页建议相反,需要维护者定哪种;有没有人在做:同上,无;怎么验证:提问式 issue,不带修改方案 |
