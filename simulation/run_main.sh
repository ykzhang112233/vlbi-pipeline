#!/bin/bash

# run python scripts and backup log files
set -euo pipefail

# directory where python will write results (should match python --out_dir)
INPUT_FILE="/groups/public_cluster/home/ykzhang/VLBI/grb_data/bl307/calibrated_data_GRB221009a-v1/GRB221009A-ba161b1.uvf"
OUT_DIR="/groups/public_cluster/home/ykzhang/VLBI/grb_data/bl307/calibrated_data_GRB221009a-v1/simulations/"
# directory to store logs
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# timestamped, unique logfile name
TS=$(date +"%Y%m%d-%H%M%S")
LOGFILE="$LOG_DIR/run-${TS}.log"

echo "Logging Python output to ${LOGFILE}"

# run the python script, capture both stdout and stderr, and also show on terminal
python sim_main.py     --input_uv  "$INPUT_FILE" \
                       --nants 8 --gain_range 0.1 --sim_times 10 \
                       --out_dir "$OUT_DIR" 2>&1 | tee "$LOGFILE"

