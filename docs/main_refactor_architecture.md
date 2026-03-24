# main.py 重构评审与迁移建议（面向当前 VLBI Pipeline）

## 现状摘要

当前主流程是典型的“**多级 flag + 串行 if 分支**”架构：

- 顶层由 `step1/step2/step3` 决定阶段是否执行。
- 每个阶段再展开为十几个子 flag（如 `load_flag`、`geo_prep_flag`、`apcal_flag`、`do_fringe_flag`）。
- 主逻辑集中在 `run_main()` 中，任务调用直接散布在大段 if 分支里。
- 配置通过 `config.py` 动态导入 Python 文件后，使用大量全局变量（`from config import *` 风格）。
- 任务实现（AIPS 指令封装）主要位于 `run_tasks.py`，本身可复用，但缺少统一调度层。

这类结构在“先跑通流程”阶段很有效，但在“稳定维护 + 升级演进”阶段会遇到明显扩展瓶颈。

---

## 当前模式是否合理（优点/问题）

### 优点

1. **可读性直观（早期）**：按处理顺序写，便于单次排障。
2. **与 AIPS 贴近**：任务参数和 AIPS 执行关系清楚，科学流程透明。
3. **手动可控**：任何步骤都能通过 flag 开关临时开/关。

### 问题

1. **flag 膨胀**：阶段逻辑与子任务开关耦合，易出现重复与矛盾状态。
2. **依赖隐式**：任务间前后依赖埋在 if 顺序里，难看出“必须先做什么”。
3. **回归困难**：改动一个 flag 可能影响多个阶段，缺少统一执行日志与可对比序列。
4. **配置不安全**：动态执行 `.py` 配置对字段合法性缺少显式校验。
5. **迁移成本高**：主流程过于集中，不利于 Python 3 清理和单元测试引入。

---

## 重构建议（P0/P1/P2）

### P0（高收益、低风险，建议先做）

#### P0-1：引入“步骤定义 + 统一执行器”，保持任务函数不变

**目的**：把“流程编排”从 `main.py` 大段 if 中抽离，先不改科学算法和 AIPS 任务实现。  
**方案简述**：新增 `pipeline_steps.py` 管“步骤定义”，新增 `pipeline_runner.py` 管执行；`run_tasks.py` 暂不大改。  
**影响文件**：

- 新增：`pipeline_main/pipeline_steps.py`
- 新增：`pipeline_main/pipeline_runner.py`
- 修改：`pipeline_main/main.py`（仅保留参数准备和 runner 调用）

**示例框架（骨架代码）**：

```python
# pipeline_main/pipeline_steps.py
class Step(object):
    def __init__(self, name, enabled, depends_on, action):
        self.name = name
        self.enabled = enabled
        self.depends_on = depends_on or []
        self.action = action


def build_steps(ctx):
    return [
        Step('load_data', ctx.flags.load_flag, [], ctx.actions.load_data),
        Step('geo_prep', ctx.flags.geo_prep_flag, ['load_data'], ctx.actions.geo_prep),
        Step('inspect', ctx.flags.inspect_flag, ['geo_prep'], ctx.actions.inspect),
        Step('apcal', ctx.flags.apcal_flag, ['inspect'], ctx.actions.apcal),
        Step('fringe_first', ctx.flags.do_fringe_flag, ['apcal'], ctx.actions.fringe_first),
    ]
```

```python
# pipeline_main/pipeline_runner.py
class PipelineRunner(object):
    def __init__(self, logger):
        self.logger = logger
        self.done = set()

    def run(self, steps):
        for step in steps:
            if not step.enabled:
                self.logger.info('Skip step: %s', step.name)
                continue
            missing = [d for d in step.depends_on if d not in self.done]
            if missing:
                raise RuntimeError('Step %s missing deps: %s' % (step.name, missing))
            self.logger.info('Start step: %s', step.name)
            step.action()
            self.done.add(step.name)
            self.logger.info('Done step: %s', step.name)
```

**工时预估**：1~2 天（不改任务函数）。  
**风险与回滚点**：

- 风险：步骤依赖写错会导致中断。
- 回滚：保留旧 `run_main_legacy()`，通过环境变量或参数切换回旧流程。

---

#### P0-2：配置读取双轨（保留旧 `.py`，新增 `YAML/JSON`）

**目的**：降低迁移风险，同时给新配置增加结构化校验入口。  
**方案简述**：`config_loader.py` 支持 `.py/.yaml/.json`，并统一输出字典对象；先做“轻校验”。  
**影响文件**：

- 新增：`pipeline_main/config_loader.py`
- 修改：`pipeline_main/config.py`（对接 loader，保留兼容）

**示例框架（骨架代码）**：

```python
# pipeline_main/config_loader.py
import os
import json

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif ext in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError('PyYAML is required for yaml config')
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    elif ext == '.py':
        data = _load_py_module_as_dict(path)
    else:
        raise ValueError('Unsupported config extension: %s' % ext)

    validate_minimal(data)
    return data


def validate_minimal(data):
    required = ['AIPS_NUMBER', 'AIPS_VERSION', 'file_path', 'file_name', 'step1', 'step2', 'step3']
    for key in required:
        if key not in data:
            raise ValueError('Missing config key: %s' % key)
```

**工时预估**：1 天。  
**风险与回滚点**：

- 风险：字段映射不齐导致旧参数取不到。
- 回滚：默认仍优先读取旧 `.py` 配置，`yaml/json` 作为可选路径。

---

#### P0-3：把 flag 组装逻辑收敛成一个对象

**目的**：避免 flag 在文件顶部散落，降低误改概率。  
**方案简述**：新增 `flags.py`，由 `step1/2/3` 推导子 flags。  
**影响文件**：

- 新增：`pipeline_main/flags.py`
- 修改：`pipeline_main/main.py`

**示例框架（骨架代码）**：

```python
# pipeline_main/flags.py
class Flags(object):
    def __init__(self, cfg):
        self.step1 = int(cfg['step1'])
        self.step2 = int(cfg['step2'])
        self.step3 = int(cfg['step3'])

        self.load_flag = 1 if self.step1 else 0
        self.listr_flag = 1 if self.step1 else 0
        self.dtsum_flag = 1 if self.step1 else 0
        self.geo_prep_flag = 1 if self.step1 else 0
        self.inspect_flag = 1 if self.step1 else 0

        self.apcal_flag = 1 if self.step2 else 0
        self.pang_flag = 1 if self.step2 else 0
        self.do_fringe_flag = 1 if self.step2 else 0

        self.do_fr_fringe_flag = 1 if self.step3 else 0
```

**工时预估**：0.5 天。  
**风险与回滚点**：

- 风险：旧变量名与新对象字段名不一致。
- 回滚：保留旧全局变量赋值，先做并行对照日志。

---

### P1（中期优化）

#### P1-1：任务注册表（Task Registry）

**目的**：把任务元信息（输入、输出、前置条件）集中管理，便于审计与自动校验。  
**方案简述**：建立 `task_registry.py`，用任务名映射到 `run_tasks.py` 的函数。  
**影响文件**：新增 `pipeline_main/task_registry.py`，修改 `main.py/pipeline_runner.py`。  
**工时预估**：2~3 天。  
**风险**：初次登记不全。  
**回滚**：runner 中允许“注册表缺失时走直调”。

#### P1-2：统一执行日志与步骤结果摘要

**目的**：支持“旧流程 vs 新流程”执行序列比对。  
**方案简述**：每步输出 `step_name/start/end/status/duration` 到单独日志或 JSON。  
**影响文件**：`pipeline_runner.py`、`logging_config.py`。  
**工时预估**：1 天。

---

### P2（长期演进）

#### P2-1：命令模式（Command）支持重试/回滚

**目的**：增强失败恢复能力。  
**方案简述**：每个步骤定义 `execute()` + `rollback()`，并支持失败重试策略。  
**工时预估**：3~5 天。  
**备注**：需在 P0 稳定后再做。

#### P2-2：后端抽象（ParselTongue backend）

**目的**：提高可测试性，便于将来替换执行后端。  
**方案简述**：增加 backend interface，现阶段仅实现 AIPS backend。  
**工时预估**：1~2 周。

---

## Python 3 升级必要性与迁移路线

### 结论

**有必要升级，并建议尽快完成“可运行基线升级”**。  
理由：

1. Python 2 已停止维护，生态依赖长期不可持续。
2. 当前仓库已有 Python 3 痕迹，但仍保留历史兼容逻辑，维护成本高。
3. 后续结构化重构（类型提示、测试、配置校验）在 Python 3 下收益更高。

### 分阶段路线

#### 短期（1 周内）

- 统一以 Python 3 环境运行主流程。
- 清理 Python2 专用兼容分支（不影响主逻辑者优先）。
- 修正打包元数据中的 Python 版本声明，避免误导安装。

#### 中期（1~2 周）

- 在 `pipeline_main` 关键模块补充类型提示（先 public 函数签名）。
- 统一日志输出，减少 `print` 混用。
- 增加最小 smoke test（配置加载 + 步骤选择 + runner 空跑）。

#### 长期（2~4 周）

- 完成任务注册表与配置 schema 化。
- 引入更严格校验（字段范围、源数量与目标数量一致性）。

---

## 分阶段实施计划（按周或按迭代）

### 第 1 周（MVP 结构改造）

1. 新增 `flags.py`、`pipeline_steps.py`、`pipeline_runner.py`。
2. `main.py` 接入 runner，但保留旧入口 `run_main_legacy()`。
3. 加入执行序列日志（至少记录 step name 与状态）。

### 第 2 周（配置与兼容）

1. 新增 `config_loader.py`，支持 `.py/.yaml/.json`。
2. 保持旧配置兼容，新增一个 YAML 示例配置。
3. 文档更新运行方式（旧路径 + 新路径）。

### 第 3 周（稳定化）

1. 建立任务注册表雏形（先覆盖 step1/step2 常用任务）。
2. 增加 smoke test 与对比脚本（新旧流程日志比对）。
3. 评估是否切换默认入口为新 runner。

---

## 验证清单与上线闸门

### 验证清单

1. **配置加载验证**：同一配置在 `.py` 与 `.yaml/.json` 下字段一致。
2. **步骤选择验证**：`step1/2/3` 组合触发的任务序列与旧流程一致。
3. **单步验证**：单独运行 `load_data`、`geo_prep`、`fringe_first` 均成功。
4. **全流程 dry-run**：不改科学参数前提下，日志序列可复现。
5. **结果一致性抽检**：抽检关键表版本和中间产物命名一致。

### 上线闸门

- 闸门1：连续 3 组历史数据通过 smoke test。
- 闸门2：新旧流程任务序列一致率达到 100%。
- 闸门3：出现异常可一键切回 legacy 入口。

---

## 建议创建/修改的文件列表

### 建议创建

- `pipeline_main/flags.py`
- `pipeline_main/pipeline_steps.py`
- `pipeline_main/pipeline_runner.py`
- `pipeline_main/config_loader.py`
- `pipeline_main/configs/example_input.yaml`

### 建议修改

- `pipeline_main/main.py`（改为轻量入口 + runner 调度）
- `pipeline_main/config.py`（改为调用 loader，保留兼容）
- `pipeline_main/logging_config.py`（增强步骤日志格式）
- `docs/usage/usage.rst`（更新运行示例与迁移说明）

---

## 3条最先落地的动作（可立即执行）

1. 在 `main.py` 中新增 `run_main_legacy()`，并写一个最小 `PipelineRunner` 入口（先只接 step1）。
2. 新建 `flags.py`，把所有子 flag 推导集中到一个对象，避免散落赋值。
3. 新建 `config_loader.py`，让旧 `.py` 配置路径保持可用，同时支持新 YAML 配置示例。

## 1条最低风险回滚策略

保留双入口：默认仍走 legacy，新增环境变量 `VLBI_PIPELINE_ENGINE=runner` 才启用新调度器；一旦发现回归，切回 legacy 无需改配置文件。