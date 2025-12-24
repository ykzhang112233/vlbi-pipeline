import pexpect
import os
import sys
import re
import time, shutil
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def init_difmap(timeout=8000000):
    """初始化 difmap 会话并返回 difmap 对象和日志文件名"""
    difmap = pexpect.spawn('difmap')
    difmap.waitnoecho
    difmap.expect('0>')
    p = re.compile(r'(difmap.log_?\d*)')
    logfile = p.findall(difmap.before.decode())[0]
    difmap.timeout = timeout
    return difmap, logfile

def cleanup_difmap(difmap, logfile):
    """关闭 difmap 会话并删除日志文件"""
    difmap.sendline('quit')
    difmap.close()
    if os.path.isfile(logfile):
        try:
            os.remove(logfile)
        except Exception:
            print(f"Can not find or delete logfile: {logfile}, will just ignore it.")
            pass
        
def setup_mapsize(difmap, freq):
    """根据频率设置地图大小"""
    if freq <= 3:
        difmap.sendline('mapsize 2048,0.4')
    elif 3.1 <= freq <= 10:
        difmap.sendline('mapsize 2048,0.2')
    elif freq >= 10.1:
        difmap.sendline('mapsize 2048,0.1')
    difmap.expect('0>')

def prepare_observation(difmap, filename,file_exname, freq):
    """准备观测：加载文件，选择偏振，设置地图大小和 UV 权重"""
    uvf_file = filename + '.' + file_exname
    print(f"Loading UV file: {uvf_file}")
    difmap.sendline('obs %s' % uvf_file)
    difmap.expect('0>')
    difmap.sendline('select i')
    difmap.expect('0>')
    setup_mapsize(difmap, freq)
    difmap.sendline('uvw 0,-2')
    difmap.expect('0>')

def getsnr_difmap(difmap):
    difmap.sendline('invert')
    difmap.expect('0>')
    difmap.sendline('print peak(flux,max)/imstat(rms)')
    difmap.expect('0>')
    # 修改正则表达式以支持科学计数法（如 7.333e-5）
    p = re.compile(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
    s = difmap.before.decode()
    print(p.findall(s))
    snr = float(p.findall(s)[0])
    difmap.sendline('print imstat(rms)')
    difmap.expect('0>')
    s2 = difmap.before.decode()
    rms = float(p.findall(s2)[0])
    difmap.sendline('print peak(x,max)')
    difmap.expect('0>')
    s3 = difmap.before.decode()
    peakx = float(p.findall(s3)[0])
    difmap.sendline('print peak(y,max)')
    difmap.expect('0>')
    s4 = difmap.before.decode()
    peaky = float(p.findall(s4)[0])
    return snr,rms,peakx,peaky

def iterative_modelfit(difmap, snr_threshold=5.5, max_iterations=12, model_type = 1):
    """迭代模型拟合：持续添加组件直到 SNR 低于阈值或达到最大迭代次数
    parm:
        difmap: the difmap progress from initi_difmap
        snr_threshold: the snr cut to decide how many iterations to go
        max_iteration: the maximum iterations. The progress will end if either of the above two parms reach the limit
        model_type: 0, delta; 1, circular gaussian; 2, elliptical gaussian
    """
    snr, rms, pkx, pky = getsnr_difmap(difmap)
    print(snr, rms, pkx, pky)
    if snr <=6: # for weak sources, force start point to (0,0)
        # if pkx >=2 or pky >=2: 
        print("weak source, set start point to (0,0)")
        pkx=0; pky=0
    nm = 0
    while snr > snr_threshold:
        if nm >= max_iterations:
            print('limit reached, stop fitting')
            break
        # help addcmp flux, v_flux, xpos, ypos, v_pos, major, v_major, ratio, v_ration, theta, v_theta, type
        if model_type not in [0,1,2]:
            print("model_type must be 0 (delta), 1 (circular gaussian), or 2 (elliptical gaussian). Using default circular gaussian (1).")
            model_type = 1
        if model_type == 0:
            difmap.sendline('addcmp 0.1,true,%f,%f,true,0,false,1,false,0,true,0' % (pkx, pky))
        elif model_type == 1:
            pkx,pky = 0,0
            difmap.sendline('addcmp 0.1,true,%f,%f,true,0.3,true,1,false,0,true,1' % (pkx, pky))
        else: # for elliptical gaussian
            difmap.sendline('addcmp 0.1,true,%f,%f,true,0.3,true,1,true,0,true,1' % (pkx, pky))
        difmap.expect('0>')
        difmap.sendline('modelfit 100')
        difmap.expect('0>', timeout=5000)
        snr, rms, pkx, pky = getsnr_difmap(difmap)
        print(snr, rms, pkx, pky)
        difmap.sendline('modelfit 100')
        difmap.expect('0>', timeout=1000)
        nm += 1
    return nm

def simple_modelfit(difmap):
    """简单模型拟合：执行一次 modelfit 命令"""
    difmap.sendline('addcmp 0.1,true,0,0,true,0.3,true,1,false,0,true,1')
    difmap.expect('0>')
    difmap.sendline('modelfit 100')
    difmap.expect('0>', timeout=5000)
    difmap.sendline('modelfit 50')
    difmap.expect('0>', timeout=1000)
                       
def read_observation(difmap,filename):
    par_file = filename + '.par'
    difmap.sendline('@ %s' % par_file)
    difmap.expect('0>')

def read_difmap_script(difmap,script_path,outname):
    difmap.sendline('@%s %s' % (script_path, outname))
    difmap.expect('Writing difmap environment.*0>',timeout=5000)

def save_df_file(difmap,filename):
    mod_file = filename
    difmap.sendline('save %s' % mod_file)
    difmap.expect('0>')
 
def parse_model_table(
        text: str, 
        default_freq=None
        ) -> pd.DataFrame:
    rows = []

    def to_float(tok: str) -> float:
        # 兼容偶尔出现的 0.123v 这种标记（这里一般不会出现在 East/North 表，但顺手处理）
        tok = tok.rstrip("v")
        try:
            return float(tok)
        except Exception:
            return np.nan

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("!") or line.startswith("-"):
            continue
        if "0>" in line:
            continue

        parts = line.split()

        # 有些 difmap 输出会带 component 编号（第一列是整数），这里做兼容
        comp_id = None
        if parts and parts[0].isdigit():
            comp_id = int(parts[0])
            parts = parts[1:]

        # East/North 那张表：最短到 theta_err 一共 15 列
        if len(parts) < 15:
            continue

        # 关键：只吃 East/North 那张表（第7列是 shape，且是纯字母，如 gauss/point）
        # 这能自动跳过你前面那行 "0.00346423v 0.141708v ... 1"
        if not (len(parts) > 6 and parts[6].isalpha()):
            continue

        row = {
            "component_id": comp_id,
            "flux_jy": to_float(parts[0]),
            "flux_err_jy": to_float(parts[1]),
            "x_arcsec": to_float(parts[2]),
            "x_err_arcsec": to_float(parts[3]),
            "y_arcsec": to_float(parts[4]),
            "y_err_arcsec": to_float(parts[5]),
            "type": parts[6],
            "ra_deg": to_float(parts[7]),
            "dec_deg": to_float(parts[8]),
            "major_fwhm_mas": round(to_float(parts[9]) * 1000, 3),
            "major_fwhm_err_mas": round(to_float(parts[10]) * 1000, 3),
            "minor_fwhm_arcsec": to_float(parts[11]),
            "minor_fwhm_err_arcsec": to_float(parts[12]),
            "theta_deg": to_float(parts[13]),
            "theta_err_deg": to_float(parts[14]),
            "freq_hz": np.nan,
            "spectral_index": np.nan,
            "spectral_index_err": np.nan,
        }

        # 如果后面真的有 freq/specIndex，就补上
        if len(parts) >= 16:
            row["freq_hz"] = to_float(parts[15])
        if len(parts) >= 17:
            row["spectral_index"] = to_float(parts[16])
        if len(parts) >= 18:
            row["spectral_index_err"] = to_float(parts[17])

        # 如果输出没带 freq，但你知道观测频率，就用 default_freq 补齐
        if (np.isnan(row["freq_hz"]) or row["freq_hz"] == 0.0) and default_freq is not None:
            row["freq_hz"] = float(default_freq)

        rows.append(row)

    df = pd.DataFrame(rows)

    # 如果 component_id 一列全是空，就删掉，表更干净
    if "component_id" in df.columns and df["component_id"].isna().all():
        df = df.drop(columns=["component_id"])

    return df

def get_model_parm(difmap):
    """
    在 difmap 里执行一条命令，并返回从命令回显到下一个提示符之间的全部文本输出。
    参数:
        difmap: difmap 进程对象
        prompt: difmap 提示符的正则表达式，默认为 "0>"
    返回:
        str: 命令输出的文本(text)
    """
    difmap.sendline('modelfit 0')
    difmap.expect('0>', timeout=100)
    out = difmap.before or b""
    print(out)
    if isinstance(out, bytes):
        out = out.decode("utf-8", errors="replace")
    out = out.replace("\r", "")

    # difmap 通常会回显你输入的命令，把第一行是命令的情况去掉
    lines = out.splitlines()
    # if lines and lines[0].strip() == cmd.strip():
    lines = lines[1:]
    return "\n".join(lines).strip()

def main(
        uvf_path:Path,
        freq:float,
        script_path:str = './single_mf_dfmp',
        debug:bool = False,
        )->pd.Series:
    filepath = Path(uvf_path)
    # copy the script in the code directory to the working directory
    file_dir = filepath.parent
    filename = filepath.stem
    shutil.copy(script_path, file_dir)
    script_name =  'single_mf_dfmp'
    file_exname = filepath.suffix.lstrip('.')
    os.chdir(file_dir)
    difmap, logfile = init_difmap()
    prepare_observation(difmap, filename,file_exname, freq)
    selection = 0
    if selection == 0:
        # simple modelfit
        print("Using simple modelfit for model fitting.")
        simple_modelfit(difmap)
    elif  selection == 1:
        filename= filename + '_scr'
        print("Using difmap script for model fitting.")
        read_difmap_script(difmap,script_name,filename)
    else:
        print("Using iterative modelfit for model fitting.")
        nm = iterative_modelfit(difmap, snr_threshold=3, max_iterations=1, model_type = 1)
        print(f"Total fitted components: {nm}")
    model_text = get_model_parm(difmap)
    print("Model fitting output:")
    print(model_text)
    df_model = parse_model_table(model_text, default_freq=None)
    print("Parsed model parameters:")
    print(df_model)
    if debug:
        print("Debug mode, will save difmap output files")
        save_df_file(difmap,filename + '_sav')
    cleanup_difmap(difmap, logfile)
    return df_model
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit model to UV data using difmap.')
    parser.add_argument('--file_path', type=str, required=True, help='Input uv file path (e.g., /path/to/TARGET.uvf)')
    args = parser.parse_args()
    filepath = Path(args.file_path)
    freq = 9  # GHz, set your frequency here
    script_path = 'single_mf_dfmp'
    df_result = main(filepath, script_path, freq)
    print("Final fitted model parameters:")
    print(df_result)
### Usage example:
## python fit_in_difmap.py --file_path ./simulation/fits_uvtest.uvf