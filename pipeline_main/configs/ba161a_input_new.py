#!/usr/bin/env python
"""
VLBI Pipeline Configuration Template
=====================================
Copy this file and modify for your observation.
Naming convention: {obs_code}_input.py (e.g., bz111cl_input.py)
"""
## Header information=========================================================
AIPS_VERSION = '31DEC25'
version_date = '2025/12/31'
geo_path='/home/ykzhang/code_repo/vlbi-pipeline/geod/'  # this path is in 4022
# ============================================================================
# (step1) BASIC SETTINGS
# ============================================================================
AIPS_NUMBER = 162  # AIPS user number
antname = 'VLBA'  # Antenna array: 'VLBA' or 'EVN'

# ============================================================================
# (step1) DATA INFORMATION
# ============================================================================
file_path = '/home/ykzhang/data/vlba/ba161/'  # Path to data directory
file_name = 'ba161a.idifits'  # Better use obs_code.idifits as name
num_files = 1  # Number of files to load

# ============================================================================
# (step2) SOURCE INFORMATION
# ============================================================================
do_quack = 1  # Flag initial data
ap_dofit = 1  # Opacity calibration: 1=all antennas, or list for specific antennas
               # ap_dofit = [-1,1,1,1,1,1,1,1,1,1] for individual antenna control
solint = 4  # Solution interval in minutes

# Source names
calsource = ['3C345']  # Calibrator for fringe fitting and bandpass
target = ['P1923+2010']  # Target source
p_ref_cal = ['P1905+1943']  # Phase reference calibrator

logfilename = file_name.split('.')[0]  # Log file name

# ============================================================================
# (step2) FRINGE FITTING CONTROL
# ============================================================================
auto_fringe = 0  # 0=manual (recommended for EVN), 1=automatic
                 # If 0, set the following parameters from step1 results
                 # if 1, the following parameters in this part will be ignored and automatically determined by the pipeline

# Manual fringe fitting parameters (when auto_fringe=0)
reference_antenna = 2
search_antennas = [8, 4, 0]
scan_for_fringe = [0,23,42,0,0,23,44,0]
av_ifs_f1 = 1  # Average IFs during fringe fitting


# ============================================================================
# (step123) MANUAL FLAGGING (can be used in all steps, but will be visuallized after step2 for checking)
# ============================================================================
do_flag = 0  # Enable manual flagging

fgantennas=[[8],[1]]

# [0] for all antennas, or list of antenna numbers (starting from 1)
# each sublist corresponds to a separate flagging round, should be consistent with other flagging parameters throughout this part (before Special quack flagging)

# Channel and IF flagging
fgbchan = [0, 0]  # Beginning channels to flag
fgechan = [0, 0]  # Ending channels to flag
fgbif = [0, 0]  # Beginning IFs to flag
fgeif = [0, 0]  # Ending IFs to flag

# Time flagging
fgtimer=[[0,14,36,36,0,15,39,0],[0,15,27,54,0,15,28,54]]
# Time ranges to flag

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
outfg = 2  # Output flag table
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Special quack flagging
sp_quack_flag = 0 # 0 mean no special quack flagging, 1 means enable special quack flagging and using the following parameters.
# This special flagging is for flagging time edges of scans and/or elevation edges, which are not well handled by the normal flagging in AIPS. 
# also, keep the elements in the following lists consistent with each other.
sp_quack_ant = [[1,2]]
sp_quack_beg = [25]
sp_quack_endb = [0]
sp_quack_el = [20]

# ============================================================================
# (step2) EVN-SPECIFIC SETTINGS (for non-VLBA data mostly) 
# ============================================================================
pipepath = '/data/VLBI/EVN/eg119/pipeline-eg119a/'  # EVN pipeline path
fgfile = ''
antfile = ''

if antname != 'VLBA':
    fgfile = pipepath + 'eg119a.uvflg'  # Flag file
    antfile = pipepath + 'eg119a.antab'  # Antenna table file


# ============================================================================
# (between step2 - 3)MAPPING CONTROL 
# ============================================================================
auto_mapping = 0  # 0=manual, 1=automatic

# ============================================================================
# (step3, optional) MANUAL ANTENNA GAIN CALIBRATION 
# ============================================================================
ant_gan_cal = 0  # Enable antenna gain calibration
pol = 'I'  # Polarization: 'I' for Stokes I only, 'LR' for both hands
# Gain correction factors per antenna per source
matxi=[[0.9,1.0,0.9,0.9,1.0,1.2,0.9,1.8,1.0],
       [0.9,1.0,0.9,0.9,1.0,1.3,0.8,1.8,1.0],
       [0.9,1.0,0.9,0.9,0.9,1.3,0.9,1.8,1.1],
       [0.9,1.0,0.9,0.9,1.0,1.2,0.9,1.5,1.0]]

# For polarization data (L and R hand)
# matxl = [[...], [...]]
# matxr = [[...], [...]]

# ============================================================================
# (Step 3) ADVANCED CALIBRATION PARAMETERS 
# ============================================================================
man_fr_file = ['J1905-v1-mod1.fits']  # Manual model files (when auto_mapping=0)
del_old_mod = True  # Delete old model before do additional fringe fitting
no_rate = 0  # Disable rate correction, dparm(9) in AIPS task FRING
rdp_parm = 0 # Whether to do zero delay/rate or phase, see AIPS task FRING manual for details (dparm(8))
dwin = 200  # Delay window
rwin = 100  # Rate window

av_ifs_f2 = 0   # whether to averege ifs during this step's fringe fitting -->cl10 (usually set to 1 if the phase-cal is weak)
av_ifs_ca1 = 1  # same with above but for calib -->cl11
# This in not used as "P" is the current version for calib -->cl11
solint_cal = 400  # the solution interval (minutes) for task CALIB "A&P"(output file is SCL11), set as large as possible if not sure
# ============================================================================
# PIPELINE CONTROL FLAGS
# ============================================================================
step1 = 0  # Data loading and initial calibration
step2 = 1  # Fringe fitting
step3 = 0  # Self-calibration and imaging
stepn = 0  # Additional post-processing

# ============================================================================
# (Step n) UV-SHIFT PARAMETERS (additional post-processing)
# ============================================================================
# Note: if you want to do averaging on IF and/or time, it is better to do UV-shift before averaging to avoid smearing effect.
# Position shifts in arcseconds (same as difmap position values)
do_uvshift_flag = 0  # Enable UV-shift (requires step3 completed)
rash = [-0.186, 0, 1.144]  # RA shift (no need to multiply by cos(dec))
decsh = [0.570, 0, 1.760]  # Dec shift
## Output _shav data with averaged among IFs

