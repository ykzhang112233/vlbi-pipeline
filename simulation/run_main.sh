#!/bin/bash

# run python scripts and backup log files
set -euo pipefail

# directory where python will write results (should match python --out_dir)
OUT_DIR="./simulation/results"
# directory to store logs
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# timestamped, unique logfile name
TS=$(date +"%Y%m%d-%H%M%S")
LOGFILE="$LOG_DIR/run-${TS}.log"

echo "Logging Python output to ${LOGFILE}"

# run the python script, capture both stdout and stderr, and also show on terminal
python sim_gain_var.py --input_uv ./simulation/fits_uvtest.uvf \
                       --nants 10 --gain_range 0.1 --sim_times 20 \
                       --out_dir "$OUT_DIR" 2>&1 | tee "$LOGFILE"

