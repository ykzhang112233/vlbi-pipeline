# VLBI Pipeline 重构评审与迁移建议 Prompt（中文模板）

你是一名资深 VLBI 数据处理与 Python 工程化顾问。请基于给定代码仓库，评估当前 `main.py` 的流程编排方式，并输出可执行的重构与迁移建议。

## 目标

围绕“`step/flag` 开关 + AIPS 任务顺序执行”的现有模式，回答以下问题：

1. 这种结构是否合理（优点与局限）？
2. 更好的结构是什么（给出分阶段改造方案）？
3. 是否有必要升级到 Python 3（给出结论、原因、风险、迁移路径）？

## 输入（调用时请补充）

- 仓库根目录：`{{repo_root}}`
- 主入口文件：`{{main_file_path}}`（默认 `pipeline_main/main.py`）
- 相关文件（可选）：
  - `{{config_file_path}}`（如 `pipeline_main/config.py`）
  - `{{task_file_path}}`（如 `pipeline_main/run_tasks.py`）
  - `{{setup_file_path}}`（如 `pipeline_main/setup.py`）
- 当前关注点（可选）：`{{focus_points}}`
- 输出文档路径（可选）：`{{output_md_path}}`（如 `docs/architecture_refactor_plan.md`）

## 约束

1. 保持科学处理逻辑不变优先，不要先改算法本身。
2. 优先给“低风险、可渐进上线”的方案。
3. 默认考虑“旧流程保留 + 新流程开关灰度上线”。
4. 配置迁移优先考虑“双轨支持”（旧 `.py` + 新 `YAML/JSON`）。
5. 所有建议需落到可执行动作（涉及文件、预计工作量、验证方式）。

## 执行步骤

1. 梳理现状
   - 识别 `step1/step2/step3/...` 与子 flag 的映射关系。
   - 标注主流程中耦合点（调度、配置加载、任务执行、日志、错误处理）。

2. 评估当前结构
   - 给出优点（例如：流程直观、排障简单）。
   - 给出缺点（例如：flag 膨胀、依赖隐式、难测试、难扩展）。

3. 输出重构建议（按优先级）
   - P0：高收益低风险（先做）。
   - P1：中期优化（可在下一迭代）。
   - P2：长期演进（插件化/并行化等）。
   - 每条建议必须包含：
     - 目的
     - 方案简述
     - 示例框架（伪代码/结构草图）
     - 影响文件
     - 工时预估
     - 风险与回滚点

4. Python 3 升级评估
   - 明确“是否必要”的结论。
   - 识别阻塞项（依赖、兼容层、打包配置）。
   - 给出分阶段迁移计划（短期/中期/长期）。

5. 验证与验收
   - 给出每阶段验证清单（配置加载、单步执行、全流程 dry-run、结果一致性）。
   - 给出切换闸门（什么时候默认启用新调度器）。

## 输出格式（必须遵守）

请输出为结构化 Markdown，包含以下章节：

1. `现状摘要`
2. `当前模式是否合理（优点/问题）`
3. `重构建议（P0/P1/P2）`
4. `Python 3 升级必要性与迁移路线`
5. `分阶段实施计划（按周或按迭代）`
6. `验证清单与上线闸门`
7. `建议创建/修改的文件列表`

并在最后附：
- `3条最先落地的动作`（可立即执行）
- `1条最低风险回滚策略`

## 风格要求

- 使用中文。
- 结论先行，条理清晰，避免空泛建议。
- 尽量给“可复制的模板/骨架代码”，但不要一次性重写整个工程。

---

## 快速调用示例

请基于以下输入执行：

- `repo_root`: `/Users/yingkangzhang/files/coding space/vlbi-pipeline`
- `main_file_path`: `pipeline_main/main.py`
- `config_file_path`: `pipeline_main/config.py`
- `task_file_path`: `pipeline_main/run_tasks.py`
- `setup_file_path`: `pipeline_main/setup.py`
- `focus_points`: `flag 编排、配置安全性、Python3迁移`
- `output_md_path`: `docs/architecture_refactor_plan.md`

输出完整评审与迁移建议。
