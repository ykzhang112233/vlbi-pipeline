#!/bin/bash

# run python scripts and backup log files
set -euo pipefail

# Filename withoud extension, e.g., "GRB221009A-bl307dx1" for "GRB221009A-bl307dx1.uvf"
FNAME="GRB221009A-bl307dx1"
# 备选名字后缀 ba161(abc)1 and bl307(bcdefg)x1
# 对应的天线数量 N_ANT[8,9,9, 9,9,10,9,11,11]
N_ANT=10
# if you set --auto_set, then FNAME and N_ANT are not used, change hard_coded parms in the sim_main.py
# --auto_set 用来自动遍历所有的 FNAME 和 N_ANT 组合，适合在设置好参数后，直接批量处理多个历元。(包含ba161abc, bl307bcdefg, 可以在sim_main.py中对应模块来指定这些组合)
N_SIM=2000
# for jk_ related mode, N_SIM is not used, the number of jk samples is determined by N_ANT or other parms in sim_main.py
SIM_MODE="gain_var"  # options: gain_var, jk_drop_ant, jk_drop_time, jk_drop_timeblock, drop_timeblock_per_ant
# 模式说明：
# gain_var: 模拟天线gain增益变化(每个天线随机10%范围浮动），适合测试增益不稳定对拟合结果的影响
# jk_drop_ant: leave-one-out方法，每次模拟丢弃一个天线，适合测试单个天线的有无对结果的影响
# jk_drop_time: leave-one-out方法，将总时间分成 10 份，每次模拟丢弃一个时间块，所有天线的时间丢弃情况一致。适合测试特定时间段对结果的影响
# jk_drop_timeblock: leave-one-out方法，逐个丢弃时间块（约1/10时间），与上面不同的是，这次是随机设置一个起点来丢弃一个1/10时间块（对所有天线一致），而不是固定的时间块，适合测试随机时间段数据缺失对结果的影响
# drop_timeblock_per_ant: 与上述类似，但是是每个天线随机丢弃一个时间块（约1/10时间），更加随机，适合测试随机时间段数据缺失对结果的影响，同时不同天线丢弃的时间段也不同
DIR="/groups/public_cluster/home/ykzhang/VLBI/grb_data/bl307/calibrated_data_GRB221009a-v1/"
INPUT_FILE="$DIR/$FNAME.uvf"
# directory where python will write results (should match python --out_dir)
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
                       --clear_temp_uv 2>&1 | tee "$LOGFILE"

