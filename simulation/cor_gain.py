import time
import sys, shutil
import os
import sys
import numpy as np
import argparse
from pathlib import Path
from astropy.io import fits
import warnings
from astropy.io.fits.verify import VerifyWarning
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

    warnings.filterwarnings("ignore", category=VerifyWarning)
    with fits.open(out_uvfits, mode="update", memmap=False) as hdul:
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
        # 计算有效可见源数量（Re/Im 非 NaN）
        vis_count = np.sum(np.isfinite(re) & np.isfinite(im))
        gains_used = np.array(gains_by_antnum) # I think this is a dummy output just to keep consisitency
        
        # 如果你想把权重也按 |fac|^2 缩放（通常系统误差MC不需要），可加：
        # data[..., 2] /= (np.abs(fac).reshape(reshape)**2)

        hdu.data = gdata  # 更新
        hdul.flush()
        print(f"Applied gains and updated UVFITS written to {out_uvfits}")

    return out_uvfits, vis_count, gains_used

def load_antenna_map_uvfits(path: str) -> dict[int, str]:
    """
    返回 {ant_num: ant_name}，自动兼容常见 UVFITS AN 表列名：
      - 天线号列：ANTENNA_NO 或 NOSTA
      - 名称列：ANNAME 或 ANNAME1（少见）
    """
    warnings.filterwarnings("ignore", category=VerifyWarning)
    with fits.open(path) as hdul:
        # 你的文件里 EXTNAME 就是 "AIPS AN"
        an_hdu = None
        for h in hdul:
            ext = (h.header.get("EXTNAME") or "").strip().upper()
            if ext == "AIPS AN" or ext.startswith("AIPS AN"):
                an_hdu = h
                break
        if an_hdu is None:
            raise RuntimeError("AN table not found (EXTNAME 'AIPS AN').")

        cols = [c.name.upper() for c in an_hdu.columns]
        data = an_hdu.data

        # 找天线号列
        ant_col_name = None
        for cand in ("ANTENNA_NO", "NOSTA"):
            if cand in cols:
                ant_col_name = cand
                break
        if ant_col_name is None:
            raise RuntimeError(f"Cannot find antenna-number column. Columns={cols}")

        # 找天线名列
        name_col_name = None
        for cand in ("ANNAME", "ANNAME1"):
            if cand in cols:
                name_col_name = cand
                break
        if name_col_name is None:
            raise RuntimeError(f"Cannot find antenna-name column. Columns={cols}")

        ant_nums = data[ant_col_name]
        names = data[name_col_name]

        # FITS 定长字符串常带空格，要 strip
        ant_map = {int(n): str(nm).strip() for n, nm in zip(ant_nums, names)}
        return ant_map
    
def jackknife_drop_antenna(
    in_file: Path,
    out_uvfits: str,
    drop_ant: int,
    zero_data: bool = False,
    ):
    """
    生成一个“等效删除某天线”的 UVFITS：
    - 保留所有表和结构不变
    - 对所有包含 drop_ant 的记录，把 weight 置 0（并可选把 Re/Im 置 0）
    drop_ant: AIPS ant number(start from 1)
    zero_data: If True, set Re/Im to 0 for dropped records (cleaner; does not affect DIFMAP because weight=0)
    """
    os.chdir(in_file.parent)
    ant_map = load_antenna_map_uvfits(in_file)
    ant_name = ant_map.get(drop_ant, f"ANT{drop_ant}")
    if os.path.exists(out_uvfits):
        print(f"Warning: output file {out_uvfits} already exists, in this mode, no need to overwrite.")
        return out_uvfits, "0 of 0", ant_name
    shutil.copyfile(in_file.name, out_uvfits)
    warnings.filterwarnings("ignore", category=VerifyWarning)
    with fits.open(out_uvfits, mode="update", memmap=False) as hdul:
        hdu = hdul[0]
        gdata = hdu.data

        baseline_arr = gdata.par("BASELINE")  # (Nblt,)
        data = gdata.data                      # [..., 3] last axis: Re, Im, Wt (通常如此)

        if data.shape[-1] != 3:
            raise ValueError(f"Unexpected DATA last axis (expect 3=Re/Im/Wt), got {data.shape}")

        # 找到需要丢弃的记录（所有包含 drop_ant 的基线-时间记录）
        mask = np.zeros(baseline_arr.shape[0], dtype=bool)
        for i, bl in enumerate(baseline_arr):
            a1, a2 = baseline_to_ants(int(bl))
            if a1 == drop_ant or a2 == drop_ant:
                mask[i] = True
        print(f"Dropping antenna {drop_ant} ({ant_name}), total records modified: ", np.sum(mask))
        n_drop = int(mask.sum())
        v_drop = f"{int(mask.sum())} of {gdata.shape[0]}"
        # the number of visibilities droped

        # 将这些记录的权重置0；可选置零 Re/Im
        # data 的第一维就是 group (Nblt)
        data[mask, ..., 2] = 0.0
        if zero_data:
            data[mask, ..., 0] = 0.0
            data[mask, ..., 1] = 0.0
        # 写个 HISTORY 方便追踪
        hdr = hdu.header
        hdr.add_history(f"JACKKNIFE: drop_ant={drop_ant}, set weight=0 for {n_drop} records.")
        if zero_data:
            hdr.add_history("JACKKNIFE: also set Re/Im=0 for dropped records.")

        hdul.flush()
        print(f"Jackknife UVFITS written to {out_uvfits}, dropped antenna {drop_ant}, total {n_drop} records affected.")

    return out_uvfits, v_drop, ant_name

def jackknife_drop_time_frac(
    in_uvfits: Path,
    out_uvfits: str,
    n_bins: int = 10,
    bin_index: int = 0,
    zero_data: bool = False,
):
    """
    用 DATE(JD, 含小数天) 作为时间轴，把观测时间跨度均分成 n_bins 段，
    丢弃第 bin_index 段（weight=0，可选 Re/Im=0），写出新 uvfits。
    # old version, only work for fix times time bins (e.g. 10 bins, so maximum 10 jk samples)
    - n_bins: 例如 10
    - bin_index: 0..n_bins-1
    """
    if not (0 <= bin_index < n_bins):
        raise ValueError(f"bin_index must be in [0, {n_bins-1}]")
    os.chdir(in_uvfits.parent)
    n_drop = f"{bin_index + 1}th out of {n_bins} time bins"
    if os.path.exists(out_uvfits):
        print(f"Warning: output file {out_uvfits} already exists, in this mode, no need to overwrite.")
        return out_uvfits, "0 of 0", n_drop
    shutil.copyfile(in_uvfits, out_uvfits)
    warnings.filterwarnings("ignore", category=VerifyWarning)
    with fits.open(out_uvfits, mode="update", memmap=False) as hdul:
        gdata = hdul[0].data
        data = gdata.data
        if data.shape[-1] != 3:
            raise ValueError(f"Unexpected DATA last axis (expect 3=Re/Im/Wt), got {data.shape}")
        cols = set([n.upper() for n in gdata.columns.names])
        if "DATE" not in cols:
            raise RuntimeError(f"No DATE column found. Available={sorted(cols)}")
        if "_DATE" not in cols:
            raise RuntimeError(f"No _DATE column found. Available={sorted(cols)}")
        
        date = np.asarray(gdata["DATE"], dtype=float)
        ddate = np.asarray(gdata["_DATE"], dtype=float)
        t = date + ddate  # 真实 JD（天）
        tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
        edges = np.linspace(tmin, tmax, n_bins + 1)  # 等时间宽度
        print(edges)
        # todo: try make random starting time and 1/10 time width for more robust jk test
        t0, t1 = edges[bin_index], edges[bin_index + 1]
        # 最后一段包含右端点避免漏掉最后一个时刻
        if bin_index == n_bins - 1:
            mask = (t >= t0) & (t <= t1)
        else:
            mask = (t >= t0) & (t < t1)
        v_drop = f"{int(mask.sum())} of {gdata.shape[0]}."
        
        # 置零权重（DIFMAP 等效丢弃）
        data[mask, ..., 2] = 0.0
        if zero_data:
            data[mask, ..., 0] = 0.0
            data[mask, ..., 1] = 0.0

        # 记录信息：把段的时间跨度也写进去（分钟）
        span_min = (t1 - t0) * 24 * 60
        hdul[0].header.add_history(
            f"JACKKNIFE: drop time-fraction bin {bin_index+1}/{n_bins}, "
            f"JD[{t0:.10f},{t1:.10f}] span~{span_min:.2f}min nrec={n_drop}"
        )
        hdul.flush()

    return out_uvfits, v_drop, n_drop

def random_drop_timeblock(
    in_uvfits: Path,
    out_uvfits: str,
    drop_frac: float = 0.10,
    edge_frac: float = 0.01,
    seed: int | None = None,
    zero_data: bool = False,
):
    """
    在有效时间区间内随机选择一个连续时间窗并丢弃（weight=0）。
    
    时间轴：t = DATE + _DATE (day)
    
    参数：
      drop_frac : 丢弃窗口占有效时长的比例（0<drop_frac<1），如 0.10
      edge_frac : 两端缓冲比例（0<=edge_frac<0.5），如 0.05 表示
                 有效区间为 [tmin+0.05*span, tmax-0.05*span]
      seed      : 随机种子（可复现）；None 则使用全局随机
      zero_data : True 则同时将 Re/Im 置 0（更干净；权重为0已足够）
    
    返回：
      (t_start, t_end, n_drop)
    """
    if not (0.0 < drop_frac < 1.0):
        raise ValueError("drop_frac must be in (0, 1)")
    if not (0.0 <= edge_frac < 0.5):
        raise ValueError("edge_frac must be in [0, 0.5)")
    
    os.chdir(in_uvfits.parent)
    rng = np.random.default_rng(seed)
    # will overwrite if exists, will error is they are the same file
    shutil.copyfile(in_uvfits, out_uvfits)

    warnings.filterwarnings("ignore", category=VerifyWarning)
    with fits.open(out_uvfits, mode="update", memmap=False) as hdul:
        gdata = hdul[0].data
        data = gdata.data
        if data.shape[-1] != 3:
            raise ValueError(f"Unexpected DATA last axis (expect 3=Re/Im/Wt), got {data.shape}")

        cols = set([n.upper() for n in gdata.columns.names])
        if "DATE" not in cols or "_DATE" not in cols:
            raise RuntimeError(f"Need DATE and _DATE columns. Available={sorted(cols)}")

        t = np.asarray(gdata["DATE"], dtype=float) + np.asarray(gdata["_DATE"], dtype=float)

        tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
        span = tmax - tmin
        if span <= 0:
            raise RuntimeError("Time span is non-positive; cannot drop time block.")

        # 定义有效区间（可去掉两端边缘）
        te0 = tmin + edge_frac * span
        te1 = tmax - edge_frac * span
        eff_span = te1 - te0
        if eff_span <= 0:
            raise RuntimeError("Effective time span is non-positive; edge_frac too large?")

        # 丢弃窗口长度
        w = drop_frac * eff_span
        if w <= 0:
            raise RuntimeError("Drop window length is non-positive.")

        # 随机选起点，保证窗口完全落在有效区间内
        t_start = rng.uniform(te0, te1 - w)
        t_end = t_start + w

        mask = (t >= t_start) & (t < t_end)
        v_drop = f"{int(mask.sum())} of {gdata.shape[0]}"
        start_frac = (t_start - tmin) / span * 100
        end_frac = (t_end - tmin) / span * 100
        n_drop = f"Dropping {start_frac:.2f}% - {end_frac:.2f}% of total time span."

        # 丢弃：权重置0（可选置0数据）
        data[mask, ..., 2] = 0.0
        if zero_data:
            data[mask, ..., 0] = 0.0
            data[mask, ..., 1] = 0.0

        hdul[0].header.add_history(
            f"RANDOM_TIME_DROP: drop_frac={drop_frac:.4f} edge_frac={edge_frac:.4f} "
            f"JD[{t_start:.10f},{t_end:.10f}) nrec={n_drop}"
        )
        hdul.flush()

    return out_uvfits, v_drop, n_drop


def main(gains_list, input_uv, out_suffix, out_dir, mode='gain_var'):

    ant_nums = len(gains_list)
    ant_dict = gen_ant_dict(ant_nums, np.array(gains_list))
    filepath = Path(input_uv)
    os.chdir(filepath.parent)
    os.makedirs(out_dir, exist_ok=True)
    out_uv = Path(out_dir) / f"{filepath.stem}_{out_suffix}{filepath.suffix}"
    ## The out_suffix will be like gcor_1, jk_dropant_1, jk_droptbin_1, and are set in the sim_main.py
    if mode == 'gain_var':
        out_uvdata, vis_count, gains_used = apply_gains_to_uvfits_by_surgery(filepath,out_uv,ant_dict)
        out_par_name = "gains"
        out_par = gains_list
        print(f"Gain correction applied and file with {out_suffix} saved to {out_uvdata}.")
    elif mode == 'jk_drop_ant':
        # here the out_suffix is like jk_dropant_1, just extract the ant number
        ant_to_drop = int(out_suffix.split('_')[-1])
        out_uvdata, vis_dropped, ant_dropped = jackknife_drop_antenna(filepath,out_uv,drop_ant=ant_to_drop,zero_data=False)
        out_par_name = "dropped_ant"
        out_par = f"dropped antenna: {ant_dropped}, dropped visibilities: {vis_dropped}"
        print(f"Jackknife (drop antenna {ant_dropped}) applied and file with {out_suffix} saved to {out_uvdata}.")
    elif mode == 'jk_drop_time':
        # here the out_suffix is like jk_droptbin_1, just extract the bin index
        bin_index = int(out_suffix.split('_')[-1]) - 1  # make it start from 0
        print(f"Dropping time bin index: {bin_index}")
        out_uvdata, vis_dropped, time_bin_dropped = jackknife_drop_time_frac(filepath,out_uv,n_bins=10,bin_index=bin_index,zero_data=False)
        # new method: random drop time block
        out_par_name = "dropped_time"
        out_par = f"dropped time bin: {time_bin_dropped}, dropped visibilities: {vis_dropped}"
        print(f"Jackknife (drop time bin {time_bin_dropped}) applied and file with {out_suffix} saved to {out_uvdata}.")
    elif mode == 'jk_drop_timeblock':
        out_uvdata, vis_dropped, time_frac_dropped = random_drop_timeblock(filepath,out_uv,drop_frac=0.10,edge_frac=0.01)
        out_par_name = "dropped_timeblock"
        out_par = f"dropped time block: {time_frac_dropped}, dropped visibilities: {vis_dropped}"
        print(f"Jackknife (drop time block {time_frac_dropped}) applied and file with {out_suffix} saved to {out_uvdata}.")
    else:
        raise ValueError(f"Unknown mode: {mode}")



    return out_uvdata, out_par_name, out_par

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply gain correction to UV data.')
    parser.add_argument('--uv_name', type=str, required=True, help='Input AIPS UV data name (e.g., TARGET.uvf)')
    parser.add_argument('--gcor_list', type=list, required=True, help='the list contain gain factor for each antenna')
    parser.add_argument('--out_suffix', type=str, required=True, help='Suffix for the output UV data file name')
    parser.add_argument('--out_dir', type=str, default='./simulations/', help='Output directory for corrected UV data')
    parser.add_argument('--mode', type=str, default='gain_var', choices=['gain_var', 'jk_drop_ant', 'jk_drop_time', 'jk_drop_timeblock'], help='Mode of operation: gain variation or jackknife')
    args = parser.parse_args()
    gains_list = args.gcor_list
    input_uv = args.uv_name
    out_suffix = args.out_suffix
    main(gains_list, input_uv, out_suffix, args.out_dir, mode=args.mode)
### Usage example:
# python cor_gain.py --uv_name "data/target.uvf" --gcor_list "[1.0, 0.95, 1.05, 1.02]" --out_suffix "gcor_1" --out_dir "./corrected_uvs/"