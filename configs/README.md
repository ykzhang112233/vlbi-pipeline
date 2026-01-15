# Configuration Files

This directory contains parameter files for different VLBI observations.

## File Naming Convention

Use the format: `{obs_code}_input.py`

Example: `bz111cl_input.py`, `ba158l1_input.py`

## Usage

When running the pipeline, specify the configuration file:

```bash
# Method 1: Using command-line argument (Recommended)
ParselTongue main.py --config configs/bz111cl_input.py

# Method 2: Using environment variable
export VLBI_CONFIG=configs/bz111cl_input.py
ParselTongue main.py
```

## Template

See `template_input.py` for a template configuration file with all available parameters.
