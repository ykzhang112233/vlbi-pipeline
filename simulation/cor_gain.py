import time
import sys
import os
import sys
import numpy as np
import argparse
import AIPS, os, math, time
from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList
from AIPSData import AIPSUVData, AIPSImage
from Wizardry.AIPSData import AIPSUVData as WAIPSUVData
sys.path.append('../vlbi-pipeline/')
from utils import gcal_apply,copy_uvdata
from run_tasks import loadfr, run_split2, loadindx


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


if __name__ == "__main__":
    gains_test = [3,3,3,3,3,3,3,3]
    ant_nums = 8
    ant_dict = gen_ant_dict(ant_nums, np.array(gains_test))
    print(ant_dict)
    out_uv = "fits_uvtest2.uvf"
    out_uvdata = apply_gains_to_uvfits_by_surgery(filepath,out_uv,ant_dict)
    
    
    
    
    
    parser = argparse.ArgumentParser(description='Apply gain correction to AIPS UV data.')
    parser.add_argument('--data_dir', type=str, help='Input AIPS UV data name (e.g., TARGET.uvf)')
    parser.add_argument('--input_fname', type=str, required=True, help='Path to input AIPS UV data file')
    parser.add_argument('--output_uvdata', type=str, help='Output AIPS UV data name after gain correction')
    # parser.add_argument('--gain_matrix', type=str, required=True, help='Path to gain matrix file (numpy format)')
    # parser.add_argument('--cluse', type=int, default=1, help='CL table number to use for gain calibration')
    # parser.add_argument('--pol', type=str, default='RRLL', help='Polarizations to apply gain correction (e.g., RRLL)')
    
    args = parser.parse_args()
    aipsver = '31DEC19'
    AIPS.userno =  3322
    antname = 'VLBA'
    # Load input UV data
    file_name = args.input_fname
    source_name = "GRB221009A"
    source_name_1 = "GRB221009A_1"
    out_name = "GRB221009A_cor"
    in_class = args.input_fname.split('.')[1]
    out_class = "uvf"
    indata = AIPSUVData(source_name, in_class, 1, 1)
    # file_name = args.data_path.split('.')[0]
    print(args.data_dir)
    print(args.input_fname)
    if indata.exists():
        print("Input UV data already exist in AIPS.")
        print("Proceeding to gain correction...")
        loadindx(args.data_dir, file_name, source_name, in_class, 1, 1, 1, 0, antname)
    else:
        # loadfr(args.data_dir,file_name,source_name,in_class,1,antname)
        loadindx(args.data_dir, file_name, source_name, in_class, 1, 1, 1, 0, antname)
    # Load gain matrix
    # gain_matrix = np.load(args.gain_matrix)
    
    # # Apply gain correction
    # gcal_apply(indata, gain_matrix, args.cluse, args.pol)
    
    # # Save corrected data to output UV data
    # copy_uvdata(indata, args.output_uvdata.split('.')[0], args.output_uvdata.split('.')[1])
    
    # print(f"Gain correction applied and saved to {args.output_uvdata}.")