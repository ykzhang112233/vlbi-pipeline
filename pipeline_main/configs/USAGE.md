# 配置文件使用指南 / Configuration Guide

## 概述 / Overview

从 2026 年版本开始，所有参数配置文件都存放在 `configs/` 目录下，与主程序代码分离，便于管理和版本控制。

Starting from the 2026 version, all parameter configuration files are stored in the `configs/` directory, separated from the main program code for better management and version control.

---

## 快速开始 / Quick Start

### 1. 创建配置文件 / Create Configuration File

```bash
# 复制模板 / Copy template
cp configs/template_input.py configs/bz111cl_input.py

# 编辑配置 / Edit configuration
vim configs/bz111cl_input.py
```

### 2. 运行管道 / Run Pipeline

```bash
# 方法1：命令行参数（推荐）/ Method 1: Command-line argument (Recommended)
ParselTongue main.py --config configs/bz111cl_input.py

# 方法2：环境变量 / Method 2: Environment variable
export VLBI_CONFIG=configs/bz111cl_input.py
ParselTongue main.py

# 方法3：默认配置 / Method 3: Default configuration
# 创建 configs/default_input.py 后直接运行
# Create configs/default_input.py then run directly
ParselTongue main.py
```

---

## 配置文件命名规范 / Naming Convention

建议使用以下命名格式：
Recommended naming format:

```
{observation_code}_input.py
```

示例 / Examples:
- `bz111cl_input.py` - 观测代码 bz111cl
- `ba158l1_input.py` - 观测代码 ba158l1
- `eg119a_input.py` - 观测代码 eg119a

---

## 目录结构 / Directory Structure

```
vlbi-pipeline/
├── configs/                    # 配置文件目录 / Configuration directory
│   ├── README.md              # 配置说明 / Configuration guide
│   ├── template_input.py      # 配置模板 / Template file
│   ├── ba158l1_input.py       # 示例配置 / Example configuration
│   ├── bz111cl_input.py       # 你的配置 / Your configuration
│   └── default_input.py       # 默认配置（可选）/ Default config (optional)
├── vlbi-pipeline/             # 主程序代码 / Main program code
│   ├── config.py              # 配置加载器 / Configuration loader
│   ├── main.py                # 主程序 / Main program
│   └── ...
└── ...
```

---

## 配置文件内容 / Configuration Content

### 必需参数 / Required Parameters

```python
# AIPS 设置 / AIPS Settings
AIPS_NUMBER = 158
antname = 'VLBA'  # or 'EVN'

# 数据信息 / Data Information
file_path = '/path/to/data/'
file_name = 'obs_code.idifits'
num_files = 1

# 源信息 / Source Information
calsource = ['4C39.25']
target = ['J0106+00']
p_ref_cal = ['P0108+0135']
```

### 可选参数 / Optional Parameters

详见 [template_input.py](template_input.py) 查看所有可用参数和说明。

See [template_input.py](template_input.py) for all available parameters and descriptions.

---

## 高级用法 / Advanced Usage

### 使用绝对路径 / Using Absolute Paths

```bash
ParselTongue main.py --config /full/path/to/configs/your_obs_input.py
```

### 使用相对路径 / Using Relative Paths

```bash
# 从项目根目录运行 / Run from project root
ParselTongue vlbi-pipeline/main.py --config configs/your_obs_input.py

# 从其他目录运行 / Run from other directory
ParselTongue main.py --config ../configs/your_obs_input.py
```

### 批处理多个观测 / Batch Processing Multiple Observations

```bash
#!/bin/bash
# process_all.sh

for config in configs/*_input.py; do
    if [ "$config" != "configs/template_input.py" ]; then
        echo "Processing $config..."
        ParselTongue main.py --config "$config"
    fi
done
```

---

## 版本控制建议 / Version Control Recommendations

### .gitignore 设置

```gitignore
# 忽略个人配置 / Ignore personal configurations
configs/*_input.py

# 保留模板和示例 / Keep templates and examples
!configs/template_input.py
!configs/README.md
!configs/.gitkeep
```

### 配置文件版本管理 / Configuration Version Management

```bash
# 为重要观测创建配置备份 / Create backup for important observations
cp configs/bz111cl_input.py configs/backups/bz111cl_input_20260115.py

# 或使用 git 标签 / Or use git tags
git add configs/bz111cl_input.py
git commit -m "Config for observation bz111cl"
git tag -a bz111cl-config -m "Configuration for bz111cl observation"
```

---

## 迁移指南 / Migration Guide

### 从旧版本迁移 / Migrating from Old Version

如果你有旧的配置文件（如 `bz111cl_input.py`）在 `vlbi-pipeline/` 目录下：

If you have old configuration files (like `bz111cl_input.py`) in the `vlbi-pipeline/` directory:

```bash
# 1. 移动到 configs 目录 / Move to configs directory
mv vlbi-pipeline/bz111cl_input.py configs/

# 2. 使用新方式运行 / Run with new method
ParselTongue main.py --config configs/bz111cl_input.py
```

旧的直接导入方式仍然支持，但不推荐：
The old direct import method still works but is deprecated:

```python
# 旧方式（不推荐）/ Old way (deprecated)
# In config.py:
import bz111cl_input as inputs

# 新方式（推荐）/ New way (recommended)
# Use --config parameter
```

---

## 故障排除 / Troubleshooting

### 找不到配置文件 / Configuration File Not Found

```
Error: Configuration file not found: configs/your_obs_input.py
```

**解决方案 / Solution:**
- 检查文件路径是否正确 / Check file path is correct
- 使用绝对路径 / Use absolute path
- 确认文件存在 / Confirm file exists: `ls -l configs/`

### 配置文件语法错误 / Configuration Syntax Error

```
Error: invalid syntax in configuration file
```

**解决方案 / Solution:**
- 检查 Python 语法 / Check Python syntax
- 确保所有列表、字典正确闭合 / Ensure all lists/dicts are properly closed
- 参考模板文件 / Refer to template file

### 导入错误 / Import Error

```
ModuleNotFoundError: No module named 'inputs'
```

**解决方案 / Solution:**
- 使用新的配置加载方式 / Use new configuration loading method
- 确保使用 `--config` 参数 / Ensure using `--config` parameter

---

## 最佳实践 / Best Practices

1. **使用描述性文件名** / Use descriptive filenames
   - ✅ `bz111cl_phase1_input.py`
   - ❌ `test.py`, `config1.py`

2. **添加注释** / Add comments
   ```python
   # 2026-01-15: Adjusted gain for antenna 3 due to weather
   matxi = [[1.0, 1.0, 0.8, ...]]
   ```

3. **保持模板更新** / Keep template updated
   - 当添加新参数时更新模板 / Update template when adding new parameters

4. **使用版本控制** / Use version control
   - 为重要配置创建 git 标签 / Create git tags for important configs

5. **分离环境配置** / Separate environment configs
   ```
   configs/
   ├── production/
   │   └── bz111cl_input.py
   ├── testing/
   │   └── bz111cl_test_input.py
   └── development/
       └── bz111cl_dev_input.py
   ```

---

## 获取帮助 / Getting Help

- 查看模板文件：[template_input.py](template_input.py)
- 查看示例配置：[ba158l1_input.py](ba158l1_input.py)
- 提交 Issue：[GitHub Issues](https://github.com/SHAO-SKA/vlbi-pipeline/issues)
