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
from run_tasks import loadfr, run_split2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply gain correction to AIPS UV data.')
    parser.add_argument('--data_path', type=str, help='Input AIPS UV data name (e.g., TARGET.uvf)')
    parser.add_argument('--output_uvdata', type=str, help='Output AIPS UV data name after gain correction')
    # parser.add_argument('--gain_matrix', type=str, required=True, help='Path to gain matrix file (numpy format)')
    # parser.add_argument('--cluse', type=int, default=1, help='CL table number to use for gain calibration')
    # parser.add_argument('--pol', type=str, default='RRLL', help='Polarizations to apply gain correction (e.g., RRLL)')
    
    args = parser.parse_args()
    aipsver = '31DEC19'
    AIPS.userno =  3322
    antname = 'VLBA'
    # Load input UV data
    source_name = "GRB221009A"
    source_name_1 = "GRB221009A_1"
    out_name = "GRB221009A_cor"
    in_class = args.data_path.split('.')[1]
    out_class = "uvf"
    indata = AIPSUVData(source_name, in_class, 1, 1)
    # file_name = args.data_path.split('.')[0]
    # print(file_name)
    print(args.data_path)
    if indata.exists():
        print("Input UV data already exist in AIPS.")
        print("Proceeding to gain correction...")
    else:
        loadfr(args.data_path,source_name,source_name_1,out_class,1,antname)

    # Load gain matrix
    # gain_matrix = np.load(args.gain_matrix)
    
    # # Apply gain correction
    # gcal_apply(indata, gain_matrix, args.cluse, args.pol)
    
    # # Save corrected data to output UV data
    # copy_uvdata(indata, args.output_uvdata.split('.')[0], args.output_uvdata.split('.')[1])
    
    # print(f"Gain correction applied and saved to {args.output_uvdata}.")