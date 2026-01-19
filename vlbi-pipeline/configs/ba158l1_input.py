#!/usr/bin/env python
"""
Example configuration for observation ba158l1
"""

# Basic settings
AIPS_NUMBER = 158
antname = 'VLBA'

# Data information
file_path = '/data/VLBI/VLBA/ba158/'
file_name = 'ba158l1.idifits'
num_files = 1

# Source information
do_quack = 1
ap_dofit = 1
solint = 4
calsource = ['4C39.25']
target = ['J0106+00']
p_ref_cal = ['P0108+0135']
logfilename = file_name.split('.')[0]

# Fringe fitting
auto_fringe = 0
reference_antenna = 8
search_antennas = [3, 2, 0]
scan_for_fringe = [1, 0, 30, 0, 1, 0, 32, 0]
av_ifs_f1 = 0

# Mapping
auto_mapping = 0
man_fr_file = ['P0108-v1-mod1.fits']

# Manual flagging
do_flag = 1
fgbchan = [0, 0]
fgechan = [0, 0]
fgbif = [2, 4]
fgeif = [2, 4]
fgantennas = [[0], [2, 10]]
fgtimer = [0]
outfg = 2

sp_quack_flag = 0
sp_quack_ant = []
sp_quack_beg = []
sp_quack_endb = []
sp_quack_el = []

# EVN settings (not used for VLBA)
pipepath = '/data/VLBI/EVN/eg119/pipeline-eg119a/'
if antname != 'VLBA':
    fgfile = pipepath + 'eg119a.uvflg'
    antfile = pipepath + 'eg119a.antab'
else:
    fgfile = ''
    antfile = ''

# Antenna gain calibration
matxi = [[1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],
         [1.1, 1.0, 0.8, 0.8, 1.0, 1.0, 0.8, 0.8, 1.2],
         [1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],
         [1.1, 1.0, 2.0, 0.8, 1.0, 1.0, 1.2, 1.0, 1.0]]
pol = 'I'
ant_gan_cal = 0

# Advanced parameters
del_old_mod = 1
no_rate = 0
av_ifs_f2 = 0
av_ifs_ca1 = 0
rdp_parm = [0, 0, 0, 0, 0, 0, 0, 0, 0]
dwin = 0
rwin = 0
solint_cal = 2

# Pipeline control
step1 = 0
step2 = 1
step3 = 1
stepn = 0

# UV-shift
rash = [-0.186, 0, 1.144]
decsh = [0.570, 0, 1.760]
do_uvshift_flag = 1
