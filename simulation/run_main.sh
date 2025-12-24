#!/bin/bash

# run python scripts and backup log files
set -euo pipefail

# directory where python will write results (should match python --out_dir)
FNAME="GRB221009A-ba161a1"
# ba161(abc)1 and bl307(bcdefg)x1
# [8,9,9, 9,9,10,9,11,11]
N_ANT=8
# if you set --auto_set, then FNAME and N_ANT are not used, change hard_coded parms in the sim_main.py
N_SIM=5
# for jk_ related mode, N_SIM is not used, the number of jk samples is determined by N_ANT or other parms in sim_main.py
SIM_MODE="jk_drop_time"  # options: gain_var, jk_drop_ant, jk_drop_time, jk_drop_timeblock

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
python sim_main.py     --input_uv  "$INPUT_FILE" --no-auto_set  \
                       --nants $N_ANT --gain_range 0.1 --sim_times $N_SIM \
                       --s_mode $SIM_MODE --out_dir "$OUT_DIR" \
                       --no-clear_temp_uv 2>&1 | tee "$LOGFILE"

