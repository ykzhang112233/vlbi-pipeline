#!/usr/bin/env python
"""
VLBI Pipeline Configuration Template
=====================================
Converted from legacy ba161c-pip.py using the bl307ex_input_new.py format.
"""

## Header information=========================================================
AIPS_VERSION = '31DEC25'
version_date = '2026/03/27'
geo_path='/home/ykzhang/code_repo/vlbi-pipeline/geod/'  # this path is in 4022
# ============================================================================
# (step1) BASIC SETTINGS
# ============================================================================
AIPS_NUMBER = 164
antname = 'VLBA'

# ============================================================================
# (step1) DATA INFORMATION
# ============================================================================
file_path = '/home/ykzhang/data/vlba/ba161/'
file_name = 'ba161c.idifits'
num_files = 1

# ============================================================================
# (step2) SOURCE INFORMATION
# ============================================================================
do_quack = 1
ap_dofit = 1
solint = 4

calsource = ['3C345']
target = ['G221009A','P1923+2010']#,'P1923+2010']
p_ref_cal = ['P1905+1943','P1905+1943']#,'P1923+2010']

logfilename = file_name.split('.')[0]

# ============================================================================
# (step2) FRINGE FITTING CONTROL
# ============================================================================
auto_fringe = 0
reference_antenna = 2
search_antennas = [8, 4, 6, 1, 0]
scan_for_fringe = [0, 21, 41, 0, 0, 21, 43, 0]
av_ifs_f1 = 1

# ============================================================================
# (step123) MANUAL FLAGGING
# ============================================================================
do_flag = 0

fgantennas = [[0], [0], [0], [0], [0], [0], [0], [0]]
fgbchan = [1, 43, 139, 171, 229, 75, 63, 85]
fgechan = [1, 120, 149, 178, 247, 88, 65, 256]
fgbif = [1, 2, 2, 2, 2, 3, 3, 4]
fgeif = [1, 2, 2, 2, 2, 3, 3, 4]
fgtimer = [[0]]
outfg = 3

sp_quack_flag = 0
sp_quack_ant = [[1, 2]]
sp_quack_beg = [25]
sp_quack_endb = [0]
sp_quack_el = [20]

# ============================================================================
# (step2) EVN-SPECIFIC SETTINGS (for non-VLBA data mostly)
# ============================================================================
pipepath = ''
fgfile = ''
antfile = ''

if antname != 'VLBA':
    fgfile = pipepath + 'ba161c.uvflg'
    antfile = pipepath + 'ba161c.antab'

# ============================================================================
# (between step2 - 3) MAPPING CONTROL
# ============================================================================
auto_mapping = 0

# ============================================================================
# (step3, optional) MANUAL ANTENNA GAIN CALIBRATION
# ============================================================================
ant_gan_cal = 1
pol = 'I'
matxi = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.15],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.1, 1.15],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.15],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.15],
]

# ============================================================================
# (Step 3) ADVANCED CALIBRATION PARAMETERS
# ============================================================================
man_fr_file = ['P1905-v1-mod1.fits','P1905-v1-mod1.fits']
del_old_mod = True
no_rate = 0
rdp_parm = 0
dwin = 1.8
rwin = 18

av_ifs_f2 = 1
av_ifs_ca1 = 1
solint_cal = 400
final_cl_ver = 9   # currently, 9 means no-smo versino and 10 for smo version, for next step's shifting
# ============================================================================
# PIPELINE CONTROL FLAGS
# ============================================================================
step1 = 0
step2 = 0
step3 = 1
stepn = 1

# ============================================================================
# (Step n) UV-SHIFT PARAMETERS (additional post-processing)
# ============================================================================
do_uvshift_flag = 1

rash = [-0.0022679,4.6266e-04]
decsh = [-0.0002765,1.2623e-04]

# ============================================================================
# Legacy parameters retained for compatibility/reference
# ============================================================================
# TECU_model = 'jplg'
# geo_path = '/home/ykzhang/Scripts/geod/'
# def_file = '/home/ykzhang/Scripts/BeSSel/def_bessel_vlbi-v2.py'
# flagver = 2
# tyver = 1
# chk_trange = [0]
# bandcal = ['3C345']
# fr_path = '/home/ykzhang/VLBA/ba161/ba161c/'
# split_seq = 1
