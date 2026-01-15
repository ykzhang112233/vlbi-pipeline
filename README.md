# vlbi-pipeline



VLBI Data Processing Pipeline.

The documents can be found [vlbi-pipeline-documents](https://vlbi-pipeline-userguide.readthedocs.io/).



## Background

Simplest VLBI data processing pipeline.


## Install

This software depends upon the following software:

- AIPS
- ParselTongue
- Obit

Find details in [usage](docs/installation/install.rst)

## Usage

### Quick Start

```bash
# 1. 创建配置文件 / Create config file
cp configs/template_input.py configs/your_obs_input.py

# 2. 编辑参数 / Edit parameters
vim configs/your_obs_input.py

# 3. 运行管道 / Run pipeline
ParselTongue main.py --config configs/your_obs_input.py
```

### Configuration Management

**New in 2026**: Configuration files are now organized in the `configs/` directory for better management.

**Running the pipeline:**

```bash
# Method 1: Using --config parameter (Recommended)
ParselTongue main.py --config configs/your_obs_input.py

# Method 2: Using environment variable
export VLBI_CONFIG=configs/your_obs_input.py
ParselTongue main.py

# Method 3: Set as default (create configs/default_input.py)
ParselTongue main.py
```

### Documentation

- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for basic usage
- **Configuration Guide**: See [configs/USAGE.md](configs/USAGE.md) for detailed configuration instructions
- **Template**: See [configs/template_input.py](configs/template_input.py) for all available parameters
- **Examples**: Run `./run_examples.sh` to see usage examples

Find more details in [usage](docs/usage/usage.rst)



## Contributing




Feel free to PR or suggestions! [Open an issue](https://github.com/SHAO-SKA/vlbi-pipeline/issues/new) or submit PRs.



## Todo
1. Logging formats; 
2. add EVN and LBA mode
3. add multiple sources mode
4. add polarization and spectraline mode



## License

[GPL © SHAO](LICENSE)
