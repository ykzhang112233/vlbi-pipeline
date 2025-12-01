#!/usr/bin/env ParselTongue
##############################################################################
# ParselTongue Script for VLBI calibration                                   #
#                                                                            #
# Reads in data, runs indxr, checks antenna tables, writes station_file.inp, #
# control_file.inp and calibrator_file.inp (for continuum data), downloads   #
# TEC-maps and EOPs, and runs TECOR and EOP corrections, runs manual         #
# phase-cal, finge-fit on geo-sources, writes out files for fitting          #
##############################################################################
##Currently non-used commands, just leave it there##
download_flag   = 0           # Download data from archive? not needed  
geo_data_nr =  0              # data file with geo data? (<0 for no geoblock)
cont        =  0              # data file with continuum data?
line        =  0              # data file with line data?  
pr_prep_flag    = 0        # Run TECOR, EOPs, ATMOS, PANG, and position shift?
#=============================================================================
###############################
from AIPS import AIPS
import os
import sys
def_file    = './vpipe_standalone_v2.py'  # Default file with main script

aipsver     = '31DEC19'


############### Input Parameters part I#############
logfile     = 'ba161a-pype.log' 

file_path   = '/data/VLBI/VLBA/ba161a1/'
geo_path    = '../geod/'
TECU_model = 'jplg'
antname     = 'VLBA'
AIPS.userno = 162
inter_flag  = 0         # interactive (1) or non-interactive (0) mode
n           = 1         # Number of UV-data files either to download, on disk
                        # or already in AIPS and to be analyzed
defdisk     = 1         # Default AIPS disk to use (can be changed later)

############ Input Parameters part I（end）#############

#############################################################################
###                  Do not change or move this part                     ####
[filename, outname, outclass]    = [range(n),range(n),range(n)]
[nfiles, ncount, doconcat]       = [range(n),range(n),range(n)]
[outdisk, flagfile, antabfile]   = [range(n),range(n),range(n)]
for i in range(n):
    [flagfile[i], antabfile[i], outdisk[i]] = ['','',defdisk]
    [nfiles[i], ncount[i], doconcat[i]]     = [0,1,-1]
#############################################################################


###############
# Input Files #
###############
#This only for single file
#print("FILE PATH =========",file_path)
filename[0] = 'ba161a.idifits'
outname[0]  = 'ba161a'
outclass[0] = 'uvdata'
nfiles[0]    = 1                  # FITLD parameter NFILES
ncount[0]    = 1                  # FITLD parameter NCOUNT
doconcat[0]  = 0                  # FITLD parameter DOCONCAT
                    # Optional parameters for each file
# outdisk[0]   = 3                  # AIPS disk for this file (if != defdisk)
#usually for EVN stations
# flagfile[0]  = 'flagfile.uvflg'   # flag file for UVFLG
# antabfile[0] = 'antabfile.antab'  # antab file for ANTAB

print("FILE NAME =========",filename[0])#,filename[1],filename[2])
print("OUT  NAME =========",outname[0])#,outname[1],outname[2])
print(sys.argv)
if (os.path.exists(outname[0])==False):
            os.mkdir(outname[0])
#################
# Control Flags #
#################
step1        = 1        #auto control of the flags in this block
 #set to 1 for automatic procedure, set 0 to enable task by task mannual checking
step2        = 1
step3        = 1
##################################
# Data preparation and first prep#
##################################

load_flag       = 0        # Load data from disk?
listr_flag      = 0        # Print out LISTR?
dtsum_flag      = 0        # Run dtsum to check antena participation?
tasav_flag      = 0        # Run tasav on original tables?
geo_prep_flag   = 0        # Run TECOR and EOP corrections? and uvflg for evn data

#get_key_flag    = 0        # Download key-file from archive
#RDBE_check      = 0        # Check Geoblock data for RDBE errors?
if step1 == 1:
    load_flag       = 1
    listr_flag      = 1
    tasav_flag      = 1
    geo_prep_flag   = 1
    dtsum_flag      = 1
    first_cal_flag  = 1        # Include ACCOR and PANG
    
    

########################## Input Parameters part II#######################
# information to fill after first prep #
#########################################################################
## single step before step 2: find the calibrator scan as possm scan   ##
## and run possm, snplt(ty) to find refantenna and fill the rest info  ##
#########################################################################
possm_scan = [0,23,42,0,0,23,44,0]
RFIck_tran = [0]
inspect_flag    = 0                  # Run possm and snplt to check and run antab for EVN data
#if inspect_flag =2, then it will check the RIFck_tran for long RFI finds, this will ned the name of the p_ref_cal, it should be run after first inepect=1 finished.
quack_flag      = 0                  # Run quack if special considerations (e.g. EVN p-ref),outputfg=2 (VLBA, 2 sec)
#=============================================================================
antnum=0
ifnum=0
infg=2
clip_outfg=2

clip_flag       = 0                  # >=1: use aclip to cut the total-power out of 0-2.5 range in the autocorrelation, generate new fg 2 table.
#if =2, use aclip to cut with specific ranges(clpmx).
#clip works on fg2
#=============================================================================
fgantennas=[[0],[0],[0],[0],[0],[0],[0],[0]]
fgbif=[1,2,2,2,2,3,3,4]
fgeif=[1,2,2,2,2,3,3,4]
fgbchan=[1,43,139,171,229,75,63,85]
fgechan=[1,120,149,178,247,88,65,256]

#print len(fgbchan),len(fgechan),len(fgbif),len(fgantennas)
#fgbchan,fgechan,fgbif,fgeif=[[0,0],[0,0],[5,7],[5,7]]
#fgantennas=[[0],[7]]

[fgtimer,outfg]=[[0],3] #outfg must >= 3

man_uvflg_flag    = 0             # Run uvflg to flag bad IFs and antennas if needed
#=============================================================================
 
refant      = 2                      # refant=0 to select refant automatically
refant_candi= [8,4,1,0]                # candidate refant for search in fringe
calsource   = ['3C345']              # calibrator        '' => automatically
mp_source   = calsource              # fringe finder     '' => automatically
mp_timera   = possm_scan             # constrain time range for fringe finder?
bandcal     = ['3C345']              # Bandpass calibrator
#target      = ['J1430+2303']           # target sourcer
#p_ref_cal   = ['P1427+2348']           # phase ref calbrator sources '' => automatically
#targets     = target + p_ref_cal

flagver     = 2               # Flag table version to be used
tyver       = 1                      # Tsys table version to be used
chk_trange  = [0]      #timerange on p_cal for possm checking
dofit=1
#dofit=  [-1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,-1]  #usually for EVN with tsys antenas   
apfive      = 1                     #This sets the aparm(5) in fringe (how to combine IFs in the global finge fitting: 0=not combine, 1=combine all, 3=combine in halves
split_seq   = 1                 #for muti pyfiles in the same username, set this differently will avoid bugs during split and fittp, if not, use 1 only.  
#=====================================================================================
tar_names=['G221009A']
pref_names=['P1905+1943']


################# Input Parameters part II (end)##################
if step2 >= 1:
    for i in range(len(tar_names)):
        target      = [tar_names[i]]         # target sourcer
        p_ref_cal   = [pref_names[i]]          # phase ref calbrator sources '' => automatically
        targets     = [target[0],p_ref_cal[0]]
        if i == 0:
            first_cal_flag  = 1        # Include ACCOR and PANG
            pr_fringe_flag  = 1        # Do manual phase cal?
            new_amp_flag    = 1       # include BPASS and ACSCL
            apcal_flag      = 1        # Do amplitude
        else:
            first_cal_flag  = 0        # Include ACCOR and PANG
            pr_fringe_flag  = 0        # Do manual phase cal?
            new_amp_flag    = 0       # include BPASS and ACSCL
            apcal_flag      = 0        # Do amplitude
        do_fringe_flag  = 1        # Do first run of fringe cal on all sources?
        plot_first_run  = 1        # DO possm and snplt to check first run result?
        #do_band_flag    = 1
        split_1_flag    = 1        # Split calibrated data in first run?
        if step2 >=2:  #To check sources that can be directly self-calibrated
            apcal_flag      = 0        # Do amplitude calibration?
            pang_flag       = 0        # Run PANG?
            pr_fringe_flag  = 0        # Do manual phase cal?

            do_fringe_flag  = 2        # Do first run of fringe cal on all sources?
            plot_first_run  = 0        # DO possm and snplt to check first run result?
            do_band_flag    = 1
            split_1_flag    = 2        # Split calibrated data in first run?
        execfile(r''+def_file)


###################### Input Parameters part III###############################
#for networks that are not well constained with tgain(e.g. EVN)
#[[cor factor for each IF for first antenna],[cor factor for each IF for 2nd antenna]...]
#1905
matxi=[[1.0,1.0,1.2,1.0,1.0,1.0,1.0,1.0],
       [1.0,1.0,1.0,1.0,1.0,1.0,0.9,1.0],
       [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
       [1.0,1.0,1.1,1.0,1.0,1.0,0.9,1.0]]

pol='I' 
snchk=4
cluse=8
do_gaincor_flag = 1   #set this and go back to step2s

##################################
# Optional inputs for fringe fit #
##################################
fr_path='/home/ykzhang/VLBA/ba161/ba161a/'
fr_file='P1905-v1-mod1.fits'
# J0157-v1-cln4.fits(2); J2300-v1-cln4.fits(0); J0004-v1-cln3(1)
[fr_nm, fr_cls, fr_dsk, fr_sq] = ['P1905-fr','CLN',1,1]
                                             # Input image to use in FRINGE:
                                             # ['NAME','CLASS',DISK,SEQUENCE]
target      = [tar_names[0]]         # target sourcer
p_ref_cal   = [pref_names[0]]          # phase ref calbrator sources '' => automatically
targets     = [target[0],p_ref_cal[0]]
snchk=4
cluse=8
dwin,rwin=[1.2,20] # in ns and mHz
smodel = [0,0]                 # SMODEL in FRING                                
solint = 4                     # SOLINT in FRING
nmaps  = 1                     # NMAPS  in FRING
no_rate1 = 0     #for first fringe(no model)#if =1, suppress rate in fringe(dparm8), if=0 fit rate as usual
nofit_rate = 1     #for 2nd fringe(with model)#if =1, suppress rate in fringe(dparm8), if=0 fit rate as usual
#ftrate=1
#apfive      = 1
sncor_timer =[0]
if step3 == 1:
    ld_fr_fringe_flag  = 1
    do_fr_fringe_flag  = 1
    do_calib_1_flag       = 1
    check_delay_rate = 1
    split_2_flag     = 1


###################################Input Parameters part III (end)####################################

###########################
# Extra step and parameters
######################uv-shift after find that the target source is > 50 mas from the phase center###############
rash=-0.00237   #in arcsec, no need to times cos(dec)
decsh=-0.00042  #in arcsec
do_uvshift_flag = 1
#######################
##################################################
#### Use parseltongue to rorate phase in a sn table
ex_step1 = 0
##################
r_factor = 0.42
fr2_file='P1923-v1-cln2.fits'
# P1923-v1-cln2.fits(2); P1905-v1-mod2.fits(0); P1905-v1-mod2.fits(1)
[fr2_nm, fr2_cls, fr2_dsk, fr2_sq] = ['P1923-fr','CLN',1,1]
#======================================================================

if ex_step1 == 1:
    ld_fr2_fringe_flag = 1
    resi_phcal_flag = 1
    sn_rotate_flag = 1
    app_rotate_flag = 1
    splt_3_flag = 1

##############################################################################
# Start main script

execfile(r''+def_file)

#
##############################################################################

