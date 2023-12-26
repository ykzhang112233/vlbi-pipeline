#!/usr/bin/env python
import sys
import numpy as np

AIPS_NUMBER = 311
antname = 'VLBA'  # Antenna order for FITLD
geo_path = '../geod/'
#file_path = sys.argv[1]
# data information
#file_path = '../data/'
file_path = '/data/VLBI/VLBA/BL309/'
#file_name = sys.argv[2]
file_name = 'bl309cpos1.idifits' #better use obs_code.idifits as name
num_files = 1 #number of files to load
#exp_path = ''
#source information#
do_quack = 1
ap_dofit = 1
ap_dofit = [-1,1,1,1,1,1,1,1,1,1]
solint = 4
calsource   = ['4C39.25']	# calibrator for fringe fitting and bandpass(if used). '' => automatically
target	    = ['J1054+5235']	# target sourcer continuum source 
p_ref_cal   = ['P1046+5354']
#please put the corresponding files in the outname[0]/
logfilename = file_name.split('.')[0]

#####################################################
auto_fringe = 0 #for automatic step connecting step1 and step2, if =0, the following parameters must be set, please refer to the results from step1. If =1, the following parameters are ignored. It is high recommanded to set 0, especially for EVN

reference_antenna = 8
search_antennas = [3,4,2,0]
scan_for_fringe = [0,17,39,0,0,17,41,0]

auto_mapping = 1  #automatic step connecting step2 and step3, if =0, the following parameters must be set, just file name end with .fits
man_fr_file = 'P1046-v1-mod1.fits'
#####################mannual flagging################################
do_flag = 1
fgbchan=[0]
fgechan=[0]
fgbif=[0]
fgeif=[0]
fgantennas=[[1]]
#print len(fgbchan),len(fgechan),len(fgbif),len(fgantennas)
#fgbchan,fgechan,fgbif,fgeif=[[0,0],[0,0],[5,7],[5,7]]
#fgantennas=[[0],[7]]

[fgtimer,outfg]=[[0],2]

#############for_EVN_data_only########################################
pipepath='/data/VLBI/EVN/eg119/pipeline-eg119a/'
#format'/data/path/'

if antname != 'VLBA':
	fgfile = pipepath+'eg119a.uvflg'
	antfile = pipepath+'eg119a.antab'
else:
	fgfile = ''
	antfile = ''

#################
# Control Flags #
#################
# set to 1 for automatic procedure, set 0 to enable task by ta sk mannual checking
step1 = 1  # auto control of the flags in this block
step2 = 0  # Auto control of the second block
step3 = 0


########DO NOT EDIT UNLESS YOU KNOW THE MEANING ##########
AIPS_VERSION = '31DEC19'
version_date = '2016/04/06'
INTER_FLAG = 0 # interactive (1) or non-interactive (0)
main_file = 'main.py'
DEF_DISKS = 1			
# FITLD parameters, for multiple input files change ncount
# file_path = sys.argv[1]
#fr_path = exp_path

TECU_model = 'jplg'
#############################################################################
###				  Do not change or move this part					 ####
defdisk=1
n = 1
[filename, outname, outclass] = [range(n), range(n), range(n)]
[nfiles, ncount, doconcat] = [range(n), range(n), range(n)]
[outdisk, flagfile, antabfile] = [range(n), range(n), range(n)]
for i in range(n):
	[flagfile[i], antabfile[i], outdisk[i]] = ['', '', defdisk]
	[nfiles[i], ncount[i], doconcat[i]] = [0, 1, 1]
#############################################################################
###############
# Input Files #
###############
# This only for single file
# print("FILE PATH =========",file_path)
filename[0] = file_name
outname[0] = file_name.split('.')[0]
outclass[0] = 'UVDATA'
nfiles[0] = 1  # FITLD parameter NFILES
ncount[0] = num_files  # FITLD parameter NCOUNT
doconcat[0] = 1  # FITLD parameter DOCONCAT
flagfile[0] = fgfile
antabfile[0] = antfile
code = ''

#note: this version no bandpass used for fringe fitting.
#### for mannual checking###########
'''
split_outcl = 'SPLIT'

max_files = 1024

outname = range(max_files)

outname[0] = filename.split(',')[0]

execfile(r''+main_file)
'''
