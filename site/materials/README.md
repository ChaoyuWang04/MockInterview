# Materials — 对外展示物料集中管理

这个目录管理网站之外的所有对外物料。仓库整体布局：

| 位置 | 内容 |
| --- | --- |
| `src/i18n/messages/{en,zh}/` | 主页内容（三个 JSON，中英成对改） |
| `hugo-blog/` | 博客 |
| `materials/resume/` | 简历（Markdown 为唯一编辑源） |
| `docs/` | 内部文档（如 new-project-playbook.md） |

## resume/

- `resume-en.md` — 英文简历（美国求职/RA 申请用）
- `resume-zh.md` — 中文简历（国内求职用）

### 维护约定

- 改简历直接改 Markdown；需要投递用的 docx/PDF 时再从 Markdown 导出（pandoc 或在线转换）
- 简历与主页（`collections.json` 的 work/projects）描述同一段经历时，注意口径一致——尤其是任职时间和头衔
- 中英文简历面向不同市场，内容允许不同步（联系方式、课程列表等本就不同），不要求逐条对齐
