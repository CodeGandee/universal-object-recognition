# 上下文目录（context）

本目录用于集中存放项目知识、开发记录与可复用资料，便于人与 AI 助手协作与溯源。

## 结构
- `design/` 架构与技术方案
- `hints/` 开发窍门与排错指引
- `instructions/` 可复用的提示词与命令模板
- `logs/` 开发会话与结果记录
- `plans/` 里程碑与实现路线
- `refcode/` 参考实现与示例代码
- `roles/` 角色化助手上下文（每个角色一个子目录）
- `summaries/` 研究/实现结论与知识总结
- `tasks/` 任务项（进行中/已完成/待办）
- `tools/` 辅助脚本与小工具

## 命名建议
- 日志：`YYYY-MM-DD_主题-结果.md`（如：`2025-11-10_pipeline-success.md`）
- how-to：`howto-...`，排错：`troubleshoot-...`，原因分析：`why-...`
- 指令/模板：`prompt-*`、`command-*`、`snippet-*`、`template-*`
- 计划：`*-plan.md` / `*-roadmap.md`

## 写作约定
- 每个文档开头建议包含“HEADER”元信息：目的/状态/日期/依赖/受众。
- 目录下放置简短 `README.md` 说明用途与放置规范。
- 仅提交文字与小型脚本，模型权重/大文件请勿入库。
