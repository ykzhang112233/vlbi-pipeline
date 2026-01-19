#!/usr/bin/env python
"""
VLBI Pipeline Configuration Template
=====================================
Copy this file and modify for your observation.
Naming convention: {obs_code}_input.py (e.g., bz111cl_input.py)
"""

# ============================================================================
# 1. BASIC SETTINGS
# ============================================================================
AIPS_NUMBER = 158  # AIPS user number
antname = 'VLBA'  # Antenna array: 'VLBA' or 'EVN'

# ============================================================================
# 2. DATA INFORMATION
# ============================================================================
file_path = '/path/to/data/'  # Path to data directory
file_name = 'obs_code.idifits'  # Better use obs_code.idifits as name
num_files = 1  # Number of files to load

# ============================================================================
# SOURCE INFORMATION
# ============================================================================
do_quack = 1  # Flag initial data
ap_dofit = 1  # Opacity calibration: 1=all antennas, or list for specific antennas
               # ap_dofit = [-1,1,1,1,1,1,1,1,1,1] for individual antenna control
solint = 4  # Solution interval in minutes

# Source names
calsource = ['4C39.25']  # Calibrator for fringe fitting and bandpass
target = ['J0106+00']  # Target source
p_ref_cal = ['P0108+0135']  # Phase reference calibrator

logfilename = file_name.split('.')[0]  # Log file name

# ============================================================================
# FRINGE FITTING CONTROL
# ============================================================================
auto_fringe = 0  # 0=manual (recommended for EVN), 1=automatic
                 # If 0, set the following parameters from step1 results

# Manual fringe fitting parameters (when auto_fringe=0)
reference_antenna = 8
search_antennas = [3, 2, 0]
scan_for_fringe = [1, 0, 30, 0, 1, 0, 32, 0]
av_ifs_f1 = 0  # Average IFs for fringe fitting

# ============================================================================
# MAPPING CONTROL
# ============================================================================
auto_mapping = 0  # 0=manual, 1=automatic
man_fr_file = ['P0108-v1-mod1.fits']  # Manual model files (when auto_mapping=0)

# ============================================================================
# MANUAL FLAGGING
# ============================================================================
do_flag = 1  # Enable manual flagging

# Channel and IF flagging
fgbchan = [0, 0]  # Beginning channels to flag
fgechan = [0, 0]  # Ending channels to flag
fgbif = [2, 4]  # Beginning IFs to flag
fgeif = [2, 4]  # Ending IFs to flag
fgantennas = [[0], [2, 10]]  # Antennas to flag

# Time flagging
fgtimer = [0]  # Time ranges to flag
outfg = 2  # Output flag table

# Special quack flagging
sp_quack_flag = 0
sp_quack_ant = []
sp_quack_beg = []
sp_quack_endb = []
sp_quack_el = []

# ============================================================================
# EVN-SPECIFIC SETTINGS (for EVN data only)
# ============================================================================
pipepath = '/data/VLBI/EVN/eg119/pipeline-eg119a/'  # EVN pipeline path

if antname != 'VLBA':
    fgfile = pipepath + 'eg119a.uvflg'  # Flag file
    antfile = pipepath + 'eg119a.antab'  # Antenna table file
else:
    fgfile = ''
    antfile = ''

# ============================================================================
# MANUAL ANTENNA GAIN CALIBRATION
# ============================================================================
# Gain correction factors per antenna per source
matxi = [[1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],
         [1.1, 1.0, 0.8, 0.8, 1.0, 1.0, 0.8, 0.8, 1.2],
         [1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],
         [1.1, 1.0, 2.0, 0.8, 1.0, 1.0, 1.2, 1.0, 1.0]]

# For polarization data (L and R hand)
# matxl = [[...], [...]]
# matxr = [[...], [...]]

pol = 'I'  # Polarization: 'I' for Stokes I only, 'LR' for both hands
ant_gan_cal = 0  # Enable antenna gain calibration (set 1 to go back to step2)

# ============================================================================
# ADVANCED CALIBRATION PARAMETERS (Step 3)
# ============================================================================
del_old_mod = 1  # Delete old model before self-cal
no_rate = 0  # Disable rate correction
av_ifs_f2 = 0  # Average IFs for second fringe fit
av_ifs_ca1 = 0  # Average IFs for calibration
rdp_parm = 0  # Rate/delay/phase parameters
dwin = 0  # Delay window
rwin = 0  # Rate window
solint_cal = 2  # Solution interval for calibration (minutes)

# ============================================================================
# PIPELINE CONTROL FLAGS
# ============================================================================
step1 = 0  # Data loading and initial calibration
step2 = 1  # Fringe fitting
step3 = 1  # Self-calibration and imaging
stepn = 0  # Additional post-processing

# ============================================================================
# UV-SHIFT PARAMETERS (Step N)
# ============================================================================
# Position shifts in arcseconds (same as difmap position values)
rash = [-0.186, 0, 1.144]  # RA shift (no need to multiply by cos(dec))
decsh = [0.570, 0, 1.760]  # Dec shift
do_uvshift_flag = 1  # Enable UV-shift (requires step3 completed)
