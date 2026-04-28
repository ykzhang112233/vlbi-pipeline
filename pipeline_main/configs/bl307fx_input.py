#!/usr/bin/env python

AIPS_NUMBER = 315
antname = 'VLBA'  # Antenna order for FITLD
geo_path = '../geod/'
#file_path = sys.argv[1]
# data information
#file_path = '../data/'
file_path = '/data/VLBI/VLBA/bl307/'
#file_name = sys.argv[2]
file_name = 'bl307f_x.idifits' #better use obs_code.idifits as name
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
#####################################################
reference_antenna = 10
search_antennas = [3,5,0]
scan_for_fringe = [0,10,54,0,0,10,56,0]
av_ifs_f1 = 0   # average ifs if is 1 duriong fring

auto_mapping = 0  #automatic step connecting step2 and step3, if =0, the following parameters must be set, just file name end with .fits
#man_fr_file = ['J1905-v1-mod1.fits']
#####################mannual flagging################################
do_flag = 1
fgbchan=[0,0]
fgechan=[0,0]
fgbif=[0,0]
fgeif=[0,0]
fgantennas=[[9],[2,5]]
fgtimer=[[0,8,36,0,0,8,47,0],[0,8,25,0,0,8,28,0]]
#print len(fgbchan),len(fgechan),len(fgbif),len(fgantennas)
#fgbchan,fgechan,fgbif,fgeif=[[0,0],[0,0],[5,7],[5,7]]
#fgantennas=[[0],[7]]
#fgtimer=[[0],[0,0,0,0,1,1,1,1,]]

sp_quack_flag = 1
# for specific antennas, do specal quack option *e.g. in HSA or other combiniton of arrays*, format similar with do_flag
# this time for EB, SC
sp_quack_ant = [[2,11]]
sp_quack_beg = [25]
sp_quack_endb = [0]
sp_quack_el = [30]

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
matxi=[[1.0,7.8,1.0,1.0,0.9,0.9,1.1,1.0,1.4,0.9,1.0],
       [1.0,7.9,0.9,1.0,0.9,0.9,1.0,1.0,1.3,1.0,1.0],
       [1.0,8.0,1.0,1.0,0.9,1.0,1.0,1.0,1.3,1.0,1.0],
       [1.0,6.6,0.9,1.0,1.0,0.9,1.0,1.0,1.3,1.0,1.0]]

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
ant_gan_cal = 1    #set this and go back to step2s
#############################################################################
######parms may be used in step3##########################################
man_fr_file = ['J1905-v1-mod1.fits']
del_old_mod = True
no_rate = 0  # if =1 suppress rate, defualt is 0
rdp_parm = 0 # whehter zero rate, delay or phase, if not familiar, set 0
dwin = 200 # the delay search window centered on 0 and in unit of nsec, set 100 if not sure
rwin = 100 # the rate search window centered on 0 and in unit of MHz, set 200 if not sure
av_ifs_f2 = 1  # whether to averege ifs in this step's fring -->cl10
av_ifs_ca1 = 1 # same with above but for calib -->cl11
## note, if you want to make trustworthy cl11, make sure the final p_cal data is save for selet i. 
## i.e. the calib will average L and R (not make it adjustable this time)

solint_cal = 400 # the solution interval for task CALIB "A&P"(output file is SCL11), set as large as possible e.g 1000 if not sure
###########################################################################
step1 = 0  # auto control of the flags in this block
step2 = 0  # Auto control of the second block
step3 = 1
stepn = 0
#############################################################################
#in stepn
#should be peak(x),peak(y) in difmap
rash=-2.56e-3  #in arcsec, no need to times cos(dec)
decsh=-3.94e-4 #in arcsec
do_uvshift_flag = 0 ###note!! this is from stepn
