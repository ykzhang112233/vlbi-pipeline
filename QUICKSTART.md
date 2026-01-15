# Quick Reference / 快速参考

## 运行管道 / Run Pipeline

```bash
# 基本用法 / Basic usage
ParselTongue main.py --config configs/your_obs_input.py

# 使用环境变量 / Using environment variable
export VLBI_CONFIG=configs/your_obs_input.py
ParselTongue main.py
```

## 创建新配置 / Create New Config

```bash
# 1. 复制模板 / Copy template
cp configs/template_input.py configs/bz111cl_input.py

# 2. 编辑参数 / Edit parameters
vim configs/bz111cl_input.py

# 3. 运行 / Run
ParselTongue main.py --config configs/bz111cl_input.py
```

## 文件结构 / File Structure

```
configs/
├── template_input.py      # 模板 / Template
├── ba158l1_input.py       # 示例 / Example
└── your_obs_input.py      # 你的配置 / Your config

vlbi-pipeline/
├── main.py                # 运行此文件 / Run this file
└── config.py              # 自动加载配置 / Auto-loads config
```

## 常见问题 / Common Issues

### 找不到配置文件 / Config not found
```bash
# 使用绝对路径 / Use absolute path
ParselTongue main.py --config /full/path/to/configs/your_input.py

# 或相对路径从正确的目录 / Or relative path from correct directory
cd /path/to/vlbi-pipeline
ParselTongue vlbi-pipeline/main.py --config configs/your_input.py
```

### 查看加载了哪个配置 / See which config is loaded
配置文件加载时会打印路径信息
The config file path will be printed when loading

## 更多信息 / More Info

- 详细文档: [configs/USAGE.md](configs/USAGE.md)
- 模板文件: [configs/template_input.py](configs/template_input.py)
- 示例脚本: [run_examples.sh](run_examples.sh)
