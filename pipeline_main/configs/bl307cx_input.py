#!/usr/bin/env python

AIPS_NUMBER = 312
antname = 'VLBA'  # Antenna order for FITLD
geo_path = '../geod/'
#file_path = sys.argv[1]
# data information
#file_path = '../data/'
file_path = '/data/VLBI/VLBA/bl307/'
#file_name = sys.argv[2]
file_name = 'bl307c_x.idifits' #better use obs_code.idifits as name
num_files = 1 #number of files to load
#exp_path = ''
#source information#
do_quack = 1
ap_dofit = 1
#ap_dofit = [-1,1,1,1,1,1,1,1,1,1] #modify this if some antenna is not suitable for opacity in apcal
solint = 4
calsource   = ['3C454.3']	# calibrator for fringe fitting and bandpass(if used). '' => automatically
target	    = ['GRB221009A']	# target sourcer continuum source 
p_ref_cal   = ['J1905+1943']
#please put the corresponding files in the outname[0]/
logfilename = file_name.split('.')[0]

#####################################################
auto_fringe = 0 #for automatic step connecting step1 and step2, if =0, the following parameters must be set, please refer to the results from step1. If =1, the following parameters are ignored. It is high recommanded to set 0, especially for EVN

reference_antenna = 8
search_antennas = [1,5,0]
scan_for_fringe = [0,20,24,0,0,20,26,0]

auto_mapping = 0  #automatic step connecting step2 and step3, if =0, the following parameters must be set, just file name end with .fits
man_fr_file = ['J1905-v2-gsmod2.fits']
#####################mannual flagging################################
do_flag = 1
fgbchan=[0,0,0]
fgechan=[0,0,0]
fgbif=[0,0,4]
fgeif=[0,0,4]
fgantennas=[[1],[1],[0]]
fgtimer=[[0,17,55,0,0,18,31,0],[0,19,57,04,0,19,58,0],[0]]
#print len(fgbchan),len(fgechan),len(fgbif),len(fgantennas)
#fgbchan,fgechan,fgbif,fgeif=[[0,0],[0,0],[5,7],[5,7]]
#fgantennas=[[0],[7]]
#fgtimer=[[0],[0,0,0,0,1,1,1,1,]]

outfg=2


#############for_EVN_data_only########################################
pipepath='/data/VLBI/EVN/eg119/pipeline-eg119a/'
#format'/data/path/'

if antname != 'VLBA':
	fgfile = pipepath+'eg119a.uvflg'
	antfile = pipepath+'eg119a.antab'
else:
	fgfile = ''
	antfile = ''
###############Mannual calibration of antenna gain##############################################################
matxi=[[4.9,1.0,1.0,1.0,1.1,1.0,0.9,1.0,1.0],
 [4.9,0.9,1.1,1.0,1.0,1.0,0.9,1.0,1.0],
 [4.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.1],
 [5.0,1.0,1.1,1.2,1.2,1.1,1.0,1.1,1.1]]


#matxl=[[1.3,1.0,1.1,0.9,1.1,1.0,1.0,1.0,0.9,1.1],
#[1.0,0.9,1.1,1.0,1.0,1.0,1.0,1.0,1.0,0.9],
#[1.0,0.9,0.9,0.9,1.0,1.0,0.9,1.0,1.0,1.1],
#[0.9,0.9,1.0,1.0,1.2,1.0,1.2,1.1,1.0,1.2]]

#matxr=[[1.0,1.0,1.0,0.9,0.8,1.0,1.1,1.0,1.0,1.0],
#[1.1,1.0,1.2,1.0,0.9,1.1,1.0,1.1,0.9,1.0],
#[1.0,0.9,0.9,0.9,1.0,1.0,1.0,1.0,1.0,1.1],
#[1.0,1.0,1.0,0.9,0.8,1.0,1.2,0.8,0.9,1.2]]

pol='I'  #if use I correction, set POL='I' and ues matxi; if use both x and l, set pol='LR' and used maxtl and matxr.
# snchk=3
# cluse=7
ant_gan_cal = 1   #set this and go back to step2s
######parms may be used in step3##########################################
del_old_mod = True
no_rate = 0  # if =1 suppress rate, defualt is 0
rdp_parm = 0 # whehter zero rate, delay or phase, if not familiar, set 0
dwin = 300 # the delay search window centered on 0 and in unit of nsec, set 100 if not sure
rwin = 200 # the rate search window centered on 0 and in unit of MHz, set 200 if not sure
solint_cal = 400 # the solution interval for task CALIB "A&P"(output file is SCL11), set as large as possible e.g 1000 if not sure
###########################################################################
step1 = 0 # auto control of the flags in this block
step2 = 0 # Auto control of the second block
step3 = 1
stepn = 0
#############################################################################
#in stepn
#should be peak(x),peak(y) in difmap
rash=-2.56e-3  #in arcsec, no need to times cos(dec)
decsh=-3.94e-4 #in arcsec
do_uvshift_flag = 0 ###note!! this is from stepn
