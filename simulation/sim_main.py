import numpy as np
import pexpect, os
from pathlib import Path
import argparse
import pandas as pd
# from simulation.fit_in_difmap import (
#     init_difmap,
#     prepare_observation,
#     getsnr_difmap,
#     cleanup_difmap,
#     get_model_parm,
#     parse_model_table,
#     )
# from simulation.cor_gain import (
#     apply_gains_to_uvfits_by_surgery,
#     gen_ant_dict,
#     )
import fit_in_difmap, cor_gain
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing



def gen_antenna_gains(
    nants: int,
    gain_range: float = 0.1,
    dist: str = "uniform",
    seed: int | None = None,
) -> np.ndarray:
    """
    返回每根天线的幅度增益因子 g_i(real>0)。

    gain_range=0.1 -> uniform: [0.9, 1.1]
                      gaussian: mean=1, sigma=gain_range/2, 严格限定在 [0.9, 1.1]
    dist: "uniform" 或 "gaussian"   
    """
    rng = np.random.default_rng(seed)
    lo, hi = 1 - gain_range, 1 + gain_range

    if dist.lower() == "uniform":
        g = rng.uniform(lo, hi, size=nants)
    elif dist.lower() == "gaussian":
        sigma = gain_range / 2
        g = 1 + rng.normal(loc=0.0, scale=sigma, size=nants)
        g = np.clip(g, a_min=lo, a_max=hi)  # 严格限定在 [lo, hi] 范围内
        g = np.round(g, decimals=3)
    else:
        raise ValueError("dist must be 'uniform' or 'gaussian'")

    return g

def clear_uv(filename):
    """删除中间文件"""
    p = Path(filename)
    if p.is_file():
        try:
            p.unlink()
            print(f"[PID {os.getpid()}] Deleted temporary file: {filename}")
        except Exception:
            print(f"[PID {os.getpid()}] Failed to delete temporary file: {filename}, will just ignore it.")
            pass


def _run_one_sim(i, filepath_str, nants, gain_range, sim_mode,out_dir_str, clear_temp_uv=True):
    """Helper to run a single simulation iteration in a worker process.

    Returns a list of record dicts for this simulation.
    """
    import os
    from pathlib import Path
    import fit_in_difmap, cor_gain

    pid = os.getpid()
    filepath = Path(filepath_str)
    out_dir = Path(out_dir_str)
    sim_dir = out_dir /f"sim_uv_pid_{pid}" / f"sim_{i+1:04d}"
    sim_dir.makedirs(exist_ok=True)

    gains = gen_antenna_gains(nants, gain_range=gain_range, dist="uniform") #, seed=None, not necessary to set seed in parallel, give pure random
    # Use process id in logging so it's easier to trace parallel runs
    print(f"[PID {pid}] Simulation {i+1}/{os.environ.get('SIM_TIMES', '?')}")
    # add: if mod start with 'jk', do jackknife by ant or time, gains_list are not necessary, only nants used, can just give a random
    # gains = np.ones(nants)  # dummy gains for jk mode
    # out_uv = cor_gain.main(
    #     gains_list=gains.tolist(),
    #     input_uv=filepath,
    #     out_suffix=f"gainvar_{i+1}", #change this if add jk
    #     out_dir=out_dir,
    # ) # add parm for mode: gain_var, jk_ant, jk_time --- duplicate this and use different output for cor_gain.main
    
    if sim_mode == 'gain_var':
        out_suffix = f"gainvar_{i+1}"
        print(f"[PID {pid}] Applying gain variation simulation with suffix: {out_suffix}")
        print(f"[PID {pid}] Gains applied: {gains}")
    elif sim_mode == 'jk_drop_ant':
        out_suffix = f"jk_dropant_{i % nants + 1}"  # cycle through antennas
        print(f"[PID {pid}] Applying jackknife drop antenna simulation with suffix: {out_suffix}")
    elif sim_mode == 'jk_drop_time':
        out_suffix = f"jk_droptbin_{i % 10 + 1}"  # cycle through 10 time bins
        print(f"[PID {pid}] Applying jackknife drop time bin simulation with suffix: {out_suffix}")
    elif sim_mode == 'jk_drop_timeblock':
        out_suffix = f"jk_droptblock_{i + 1}" 
        print(f"[PID {pid}] Applying jackknife drop time block simulation with suffix: {out_suffix}")
    else:
        raise ValueError("sim_mode must be 'gain_var', 'jk_drop_ant', 'jk_drop_time', or 'jk_drop_timeblock'")
    out_uv, outparm_name, outparm = cor_gain.main(
        gains_list=gains.tolist(), 
        input_uv=filepath, 
        out_suffix=out_suffix, 
        out_dir=sim_dir,  # specify sim_dir to avoid conflicts, old is out_dir
        mode=sim_mode)
    
    do_debug = False
    if not clear_temp_uv:
        do_debug =  True
    df_model = fit_in_difmap.main(
        uvf_path=out_uv,
        freq=9,
        debug=do_debug,  # set true if needed
    )

    recs = []
    for rec in df_model.to_dict(orient='records'):
        # rec['gains'] = ','.join([f"{g:.2f}" for g in gains])
        rec[outparm_name] = outparm
        rec['simulation_id'] = i + 1
        recs.append(rec)

    # remove temporary uv if produced
    if clear_temp_uv:
        clear_uv(out_uv)

    return recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate gain variations and fit models in Difmap.")
    parser.add_argument('--input_uv', type=str, required=True, help='Input uv file path (e.g., /path/to/TARGET.uvf)')
    parser.add_argument('--nants', type=int, default=10, help='Number of antennas')
    # Boolean flags: provide both enable/disable forms for robust CLI parsing
    parser.add_argument('--auto_set', dest='auto_set', action='store_true', help='Run with hardcoded settings for certain experiments')
    parser.add_argument('--no-auto_set', dest='auto_set', action='store_false', help='Disable hardcoded settings for experiments')
    parser.set_defaults(auto_set=False)
    parser.add_argument('--gain_range', type=float, default=0.1, help='Gain variation range (e.g., 0.1 for ±10%)')
    parser.add_argument('--sim_times', type=int, default=10, help='Number of simulation times')
    parser.add_argument('--s_mode', type=str, default='gain_var', choices=['gain_var', 'jk_drop_ant', 'jk_drop_time', 'jk_drop_timeblock'], help='Simulation mode: gain variation or jackknife')
    parser.add_argument('--out_dir', type=str, default='./simulations/', help='Prefix for output directory for simulation results')
    parser.add_argument('--clear_temp_uv', dest='clear_temp_uv', action='store_true', help='Clear temporary uv files after simulation')
    parser.add_argument('--no-clear_temp_uv', dest='clear_temp_uv', action='store_false', help='Do not clear temporary uv files after simulation')
    parser.set_defaults(clear_temp_uv=True)
    args = parser.parse_args()
    filepath = Path(args.input_uv)
    file_name = filepath.stem
    nants = args.nants
    auto = args.auto_set
    clear_uvs = args.clear_temp_uv
    if auto: # change this hardcode setting if use --auto_set
        print("Running with hardcoded auto settings for experiments.")
        epcoch_list = ['GRB221009A-ba161a1', 'GRB221009A-ba161b1', 'GRB221009A-ba161c1',
                       'GRB221009A-bl307bx1', 'GRB221009A-bl307cx1', 'GRB221009A-bl307dx1',
                        'GRB221009A-bl307ex1', 'GRB221009A-bl307fx1', 'GRB221009A-bl307gx1']
        nant_list = [8,9,9,9,9,10,9,11,11]
        list_len = len(epcoch_list)
    else:
        epcoch_list = [file_name]
        nant_list = [nants]
        list_len = 1
    gain_range = args.gain_range
    sim_times = args.sim_times
    sim_mode = args.s_mode
    pid = os.getpid()

    # Parallel execution using processes. Each worker runs one simulation iteration.
    # WARNING: ensure `cor_gain.main` and `fit_in_difmap.main` are safe to run concurrently
    # (no shared temp files). We pass strings/paths to workers to avoid cwd issues.
    for idx in range(list_len):
        nants = nant_list[idx]
        filepath = Path(args.input_uv.replace(file_name, epcoch_list[idx]))
        print(f"Starting {sim_mode} simulations for epoch {epcoch_list[idx]} with {nants} antennas.")
        out_dir = Path(args.out_dir) / epcoch_list[idx]
        os.chdir(filepath.parent)
        os.makedirs(out_dir, exist_ok=True)
        records = []
        if sim_mode.startswith('jk'):
            # in jk mode, sim_times is determined by nants or other parms, here just set a large number to cover all
            if sim_mode == 'jk_drop_ant':
                print(f"updating sim_times to number of antennas: {nants} for jk_drop_ant mode.")
                if sim_times <= nants:
                    print(f"Note: sim_times {sim_times} is less than nants {nants}, updating to nants for at least one drop per antenna.")
                    sim_times = nants  # drop each antenna once
            elif sim_mode == 'jk_drop_time':
                print(f"updating sim_times to 10 for jk_drop_time mode.")
                if sim_times <= 10:
                    print(f"Note: sim_times {sim_times} is less than 10, updating to 10 for at least one drop per time bin.")
                    sim_times = 10  # drop each time bin once

        print(f"Simulation mode: {sim_mode}, will run {sim_times} simulations per epoch.")
        max_workers = min(sim_times, os.cpu_count() or 5)
        os.environ['SIM_TIMES'] = str(sim_times)

        if max_workers <= 1:
            # fallback to serial execution
            for i in range(sim_times):
                records.extend(_run_one_sim(i, str(filepath), nants, gain_range,sim_mode, str(out_dir), clear_uvs))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(_run_one_sim, i, str(filepath), nants, gain_range, sim_mode, str(out_dir), clear_uvs): i for i in range(sim_times)}
                for fut in as_completed(futures):
                    recs = fut.result()
                    records.extend(recs)
        df_all = pd.DataFrame.from_records(records)
        out_csv_name = f"simulated_source_parms_{epcoch_list[idx]}_{sim_mode}_{sim_times}_pid{pid}.csv"
        output_csv = out_dir / out_csv_name
        # rename if necessary to avoid overwrite
        if output_csv.is_file():
            count = 1
            while True:
                new_name = output_csv.with_name(f"{output_csv.stem}_v{count}{output_csv.suffix}")
                if not new_name.is_file():
                    output_csv = new_name
                    break
                count += 1
        df_all.to_csv(output_csv, index=False)
        print(f"All simulation results saved to {output_csv}.")
### Usage example:
## python sim_gain_var.py --input_uv ./simulation/fits_uvtest.uvf --nants 10 --gain_range 0.1 --sim_times 20 --out_dir ./simulation/results/
## python sim_gain_var.py --input_uv ./simulation/GRB221009A-ba161a1.uvf 
#                         --no-auto_set  --gain_range 0.1 
#                         --s_mode jk_drop_ant
#                         --sim_times 50 --out_dir ./simulation/results/
#                         --no-clear_temp_uv

