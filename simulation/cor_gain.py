import time
import sys, shutil
import os
import sys
import numpy as np
import argparse
from pathlib import Path
from astropy.io import fits

# Note: this single script file read the gain factor array/matrix and 
#      apply the gain factor to the uvfits data by surgery (directly modify the vis data in uvfits file)
#      input: original uvfits file, gain factor array/matrix
#      output: uvfits file after gain application (numbers)

def gen_ant_dict(
        ant_nums: int, 
        gains: np.ndarray
        ) -> dict:
    """
    将天线号数组和增益数组转换为字典形式
    Parameters
    ----------
    ant_nums : int, number of antenna
        天线数量。
    gains : np.ndarray, shape (Nants_data,)
        每个天线的增益因子数组。
    Returns
    -------
    gain_dict : dict
        天线号到增益因子的映射字典。
    """
    if ant_nums != gains.shape[0]:
        raise ValueError(f"ant_nums {ant_nums} != gains shape {gains.shape}")
    ant_list = np.arange(1, ant_nums + 1)
    gain_dict = {ant: gain for ant, gain in zip(ant_list, gains)}
    return gain_dict


def baseline_to_ants(baseline):
    """
    AIPS UVFITS baseline 编码:baseline = 256*ant1 + ant2
    """
    ant1 = baseline // 256
    ant2 = baseline % 256
    # print(f"Baseline {baseline} -> Antennas ({ant1}, {ant2})")
    return int(ant1), int(ant2)

def apply_gains_to_uvfits_by_surgery(
        in_file: Path,
        out_uvfits: str,
        gains_by_antnum: np.ndarray, 
        conj_second=False):
    """
    复制原文件 -> 只修改 vis 的 (Re, Im)；其它表（AN/SU/...）原样保留
    params:
        in_file: input uvf file path
        out_uvfits: output uv file name, will be in the same dir with input file
        gains_by_antnum: dict {ant_number: gain} 或者 array（用天线号做索引）
        conj_second: 若你未来做 complex gain：fac = g_i * conj(g_j)
    """
    os.chdir(in_file.parent)
    shutil.copyfile(in_file.name, out_uvfits)

    with fits.open(out_uvfits, mode="update", memmap=True) as hdul:
        # UVFITS visibilities 通常在 PRIMARY 的 random groups 里
        hdu = hdul[0]
        gdata = hdu.data  # GroupData

        # group parameters: UU, VV, WW, BASELINE, DATE 等
        baseline_arr = gdata.par("BASELINE")  # shape (Nblts,)
        print("Baseline array shape:", baseline_arr.shape)
        # vis 数据通常在 gdata.data 里，形状因数据而异
        # 常见是 (Nblts, 1, Nchan, Npol, 3) 或 (Nblts, Nchan, Npol, 3)
        prim_data = gdata.data

        # 找到最后一维的 Re/Im/Wt: [..., 0]=Re, [...,1]=Im, [...,2]=Wt
        if prim_data.shape[-1] != 3:
            raise ValueError(f"Unexpected DATA last axis (expect 3=Re/Im/Wt), got {prim_data.shape}")

        # 构造每条记录的增益因子
        fac = np.empty(baseline_arr.shape[0], dtype=np.complex128)
        for i, bl in enumerate(baseline_arr):
            a1, a2 = baseline_to_ants(int(bl))
            g1 = gains_by_antnum[a1]
            g2 = gains_by_antnum[a2]
            if conj_second:
                fac[i] = g1 * np.conj(g2)
            else:
                fac[i] = g1 * g2

        # 把 fac broadcast 到 DATA 的所有频率/极化维
        # 先把 Re/Im 取出来合成 complex，再乘，再写回 Re/Im
        re = prim_data[..., 0]
        im = prim_data[..., 1]
        vis = re + 1j * im

        # fac 需要 reshape 到 (Nblts, 1, 1, 1, ...) 以广播
        # 这里做一个通用 reshape：前面是 Nblts，后面补 1
        reshape = (fac.shape[0],) + (1,) * (vis.ndim - 1)
        vis *= fac.reshape(reshape)

        prim_data[..., 0] = vis.real
        prim_data[..., 1] = vis.imag

        # 如果你想把权重也按 |fac|^2 缩放（通常系统误差MC不需要），可加：
        # data[..., 2] /= (np.abs(fac).reshape(reshape)**2)

        hdu.data = gdata  # 更新
        hdul.flush()
        print(f"Applied gains and updated UVFITS written to {out_uvfits}")

    return out_uvfits

def main(gains_list, input_uv, out_suffix, out_dir):

    ant_nums = len(gains_list)
    ant_dict = gen_ant_dict(ant_nums, np.array(gains_list))
    filepath = Path(input_uv)
    os.chdir(filepath.parent)
    os.makedirs(out_dir, exist_ok=True)
    out_uv = Path(out_dir) / f"{filepath.stem}_{out_suffix}{filepath.suffix}"
    out_uvdata = apply_gains_to_uvfits_by_surgery(filepath,out_uv,ant_dict)
    print(f"Gain correction applied and file with {out_suffix} saved to {out_uvdata}.")

    return out_uvdata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply gain correction to UV data.')
    parser.add_argument('--uv_name', type=str, required=True, help='Input AIPS UV data name (e.g., TARGET.uvf)')
    parser.add_argument('--gcor_list', type=list, required=True, help='the list contain gain factor for each antenna')
    parser.add_argument('--out_suffix', type=str, required=True, help='Suffix for the output UV data file name')
    parser.add_argument('--out_dir', type=str, default='./simulations/', help='Output directory for corrected UV data')
    args = parser.parse_args()
    gains_list = args.gcor_list
    input_uv = args.uv_name
    out_suffix = args.out_suffix
    main(gains_list, input_uv, out_suffix)
### Usage example:
# python cor_gain.py --uv_name "data/target.uvf" --gcor_list "[1.0, 0.95, 1.05, 1.02]" --out_suffix "gcor_1"