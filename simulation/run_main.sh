#!/bin/bash

# run python scripts and backup log files
set -euo pipefail

# directory where python will write results (should match python --out_dir)
FNAME="GRB221009A-bl307gx1"
# ba161(abc)1 and bl307(bcdefg)x1
N_ANT=11
# a1,b1,c1  b,c,d ,e, f, g
# 8,9,9,    9,9,10,9,11,11
# if you set --auto_set, then FNAME and N_ANT are not used, change hard_coded parms in the sim_main.py
N_SIM=1000
DIR="/groups/public_cluster/home/ykzhang/VLBI/grb_data/bl307/calibrated_data_GRB221009a-v1/"
INPUT_FILE="$DIR/$FNAME.uvf"
OUT_DIR="$DIR/simulations/"
# directory to store logs
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# timestamped, unique logfile name
TS=$(date +"%Y%m%d-%H%M%S")
LOGFILE="$LOG_DIR/run-${TS}.log"

echo "Logging Python output to ${LOGFILE}"

# run the python script, capture both stdout and stderr, and also show on terminal
python sim_main.py     --input_uv  "$INPUT_FILE" --auto_set  \
                       --nants $N_ANT --gain_range 0.1 --sim_times $N_SIM \
                       --out_dir "$OUT_DIR" \
                       --no-clear_temp_uv 2>&1 | tee "$LOGFILE"

