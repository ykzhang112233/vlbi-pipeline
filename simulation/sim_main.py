import numpy as np
import pexpect, os
from pathlib import Path
import argparse
import pandas as pd
from simulation.fit_in_difmap import (
    init_difmap,
    prepare_observation,
    getsnr_difmap,
    cleanup_difmap,
    get_model_parm,
    parse_model_table,
    )
from simulation.cor_gain import (
    apply_gains_to_uvfits_by_surgery,
    gen_ant_dict,
    )
import fit_in_difmap, cor_gain



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
    else:
        raise ValueError("dist must be 'uniform' or 'gaussian'")

    return g

def clear_uv(filename):
    """删除中间文件"""
    p = Path(filename)
    if p.is_file():
        p.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate gain variations and fit models in Difmap.")
    parser.add_argument('--input_uv', type=str, required=True, help='Input uv file path (e.g., /path/to/TARGET.uvf)')
    parser.add_argument('--nants', type=int, default=10, help='Number of antennas')
    parser.add_argument('--gain_range', type=float, default=0.1, help='Gain variation range (e.g., 0.1 for ±10%)')
    parser.add_argument('--sim_times', type=int, default=10, help='Number of simulation times')
    parser.add_argument('--out_dir', type=str, default='./simulation/', help='Output directory for simulation results')
    args = parser.parse_args()
    filepath = Path(args.input_uv)
    nants = args.nants
    gain_range = args.gain_range
    sim_times = args.sim_times
    out_dir = Path(args.out_dir)

    os.chdir(filepath.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(sim_times):
        gains = gen_antenna_gains(nants, gain_range=gain_range, dist="uniform", seed=i)
        print(f"Simulation {i+1}/{sim_times}, Generated Gains: {gains}")
        out_uv = cor_gain.main(
            gains_list=gains.tolist(),
            input_uv=filepath,
            out_suffix=f"gainvar_{i+1}"
        )
        df_model = fit_in_difmap.main(
            uvf_path=out_uv,
            freq=2.3
        )
        # df_model['simulation_id'] = i + 1
        # records.append(df_model)
        for rec in df_model.to_dict(orient='records'):
            rec['simulation_id'] = i + 1
            records.append(rec)
    df_all = pd.DataFrame.from_records(records)
    output_csv = out_dir / "simulated_source_parms.csv"
    df_all.to_csv(output_csv, index=False)
    print(f"All simulation results saved to {output_csv}.")
### Usage example:
## python sim_gain_var.py --input_uv ./simulation/fits_uvtest.uvf --nants 10 --gain_range 0.1 --sim_times 20 --out_dir ./simulation/results/
