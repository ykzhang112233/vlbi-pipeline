#!/usr/bin/env ParselTongue

import time
import sys
#import pathlib

import os
from AIPS import AIPS
import logging
import argparse
from config import AIPS_VERSION, AIPS_NUMBER, INTER_FLAG, DEF_DISKS, step1, step2, step3 #, split_outcl, antname
from make_utils import *
from run_tasks import *
from get_utils import *
from check_utils import *
from plot_utils import *
from utils import *

# Init setting
aipsver = AIPS_VERSION
AIPS.userno = AIPS_NUMBER
inter_flag = INTER_FLAG
step1 = step1
step2 = step2
step3 = step3
antname = antname

# Setting the parameters
parser = argparse.ArgumentParser(description="VLBI pipeline")
parser.add_argument('--aipsnumber', metavar='aips number',
                    type=int, nargs='+', help='the AIPS number <keep only>')
parser.add_argument('--fitsfile', metavar='fits file',
                    type=str, nargs='+', help='files file name')
parser.add_argument('--filepath', metavar='file path',
                    type=str, nargs='+', help='files path')
#parser.add_argument('-p', '--file-path', type=pathlib.Path, default='/data/VLBI/VLBA/', help='the data path of fits file')
#parser.add_argument('-i', '--image-path', type=pathlib.Path, default='/data/VLBI/VLBA/images/', help='the data path of image file')
parser.add_argument('-o', '--output-filename',
                    default='demo', help='the output file name')
parser.add_argument('--step1', type=int, default=0, help='VLBI pipeline step1')
parser.add_argument('--step2', type=int, default=0, help='VLBI pipeline step2')
parser.add_argument('--step3', type=int, default=0, help='VLBI pipeline step3')

'''
args = parser.parse_args()
print(args)

file_path = args.filepath
fitsname = args.fitsfile
step1 = args.step1 
step2 = args.step2
step3 = args.step3
print(file_path)
print(fitsname)
print(step1)
print(step2)
print(step3)
'''
global line
line = 0

def run_main(logfile):

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logfile+'-test',
                        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    if os.path.exists('logs'):
        logging.info("<< Start VLBI-pipeline >>")
        logging.info("Commanding : %s ", sys.argv)
    else:
        os.mkdir('logs')
        logging.info("<< Start VLBI-pipeline >>")

    #AIPS.log = logfile+'-test'

    n = DEF_DISKS
    defdisk = 1  # Default AIPS disk to use (can be changed later)

    #############################################################################
    ###                  Do not change or move this part                     ####
    [filename, outname, outclass] = [range(n), range(n), range(n)]
    [nfiles, ncount, doconcat] = [range(n), range(n), range(n)]
    [outdisk, flagfile, antabfile] = [range(n), range(n), range(n)]
    for i in range(n):
        [flagfile[i], antabfile[i], outdisk[i]] = ['', '', defdisk]
        [nfiles[i], ncount[i], doconcat[i]] = [0, NCOUNT, -1]

    #filename[0] = 'BZ064A.idifits'
    #outname[0] = 'bz064a'
    #filename[0] = sys.argv[2]
    #fitsname = parser.fitsfile
    fitsname = sys.argv[2]
    filename[0] = fitsname
    #parms_filename = filename[0]+'-parms.txt'
    parms_filename = 'parms.txt'
    if os.path.exists(fitsname):
        print('Folder exists')
    else:
        os.mkdir(fitsname)
    outname[0] = fitsname.split('.')[0] 
    outclass[0] = 'UVDATA'
    nfiles[0] = 1  # FITLD parameter NFILES
    ncount[0] = NCOUNT  # FITLD parameter NCOUNT
    doconcat[0] = 1  # FITLD parameter DOCONCAT
    # Optional parameters for each file
    # outdisk[0]   = 3                  # AIPS disk for this file (if != defdisk)
    # usually for EVN stations
    # flagfile[0]  = 'es094.uvflg'   # flag file for UVFLG
    # antabfile[0] = 'es094.antab'  # antab file for ANTAB

    logging.info('#############################################')
    logging.info('### Using definition file from %s ###', version_date)
    logging.info('### Using AIPS Version %s ###',  aipsver)
    logging.info('#############################################')

    debug = 1
    #n = DEF_DISKS

    try:
        debug = debug
    except:
        debug = 0

    try:
        if inter_flag == 0:
            print
            'Running in non-interactive mode.'
        else:
            print
            'Running in interactive mode.'
    except:
        inter_flag = 1
        print
        'Running in interactive mode.'

    try:
        if split_outcl == '':
            split_outcl = 'SPLIT'
        else:
            if len(split_outcl) > 6:
                split_outcl = split_outcl[0:6]
                logging.info('################################################' )
                logging.warning('split_outcl longer than 6 characters. Truncating ')
                logging.warning('it to:  %s ' ,split_outcl )
                logging.info('################################################' )
    except:
        split_outcl = 'SPLIT'

    ##############################################################################
    #####                 Set default parameters, if not set                 #####
    ##############################################################################
    if 'ld_fr_fringe_flag' in locals() and globals():
        pass
    else:
        ld_fr_fringe_flag = 0
    if 'do_fr_fringe_flag' in locals() and globals():
        pass
    else:
        do_fr_fringe_flag = 0
    if 'do_calib_1_flag' in locals() and globals():
        pass
    else:
        do_calib_1_flag = 0
    if 'check_delay_rate' in locals() and globals():
        pass
    else:
        check_delay_rate = 0
    if 'split_2_flag' in locals() and globals():
        pass
    else:
        split_2_flag = 0

    if 'apcal_flag' in locals() and globals():
        pass
    else:
        apcal_flag = 0
    if 'pang_flag' in locals() and globals():
        pass
    else:
        pang_flag = 0
    if 'do_fringe_flag' in locals() and globals():
        pass
    else:
        do_fringe_flag = 0
    if 'plot_first_run' in locals() and globals():
        pass
    else:
        plot_first_run = 0
    if 'split_1_flag' in locals() and globals():
        pass
    else:
        split_1_flag = 0

    if 'pr_fringe_flag' in locals() and globals():
        pass
    else:
        pr_fringe_flag = 0
    if 'delzn_flag' in locals() and globals():
        pass
    else:
        delzn_flag = 0
    if 'restore_fg_flag' in locals() and globals():
        pass
    else:
        restore_fg_flag = 0
    if 'restore_su_flag' in locals() and globals():
        pass
    else:
        restore_su_flag = 0
    if 'do_gaincor_flag' in locals() and globals():
        pass
    else:
        do_gaincor_flag = 0
    if 'split_flag' in locals() and globals():
        pass
    else:
        split_flag = 0
    if 'ma_imagr_flag' in locals() and globals():
        pass
    else:
        ma_imagr_flag = 0
    if 'co_imagr_flag' in locals() and globals():
        pass
    else:
        co_imagr_flag = 0
    if 'cube_imagr_flag' in locals() and globals():
        pass
    else:
        cube_imagr_flag = 0
    if 'fr_nm' in locals() and globals():
        pass
    else:
        fr_nm = ''
    if 'fr_cls' in locals() and globals():
        pass
    else:
        fr_cls = ''
    if 'fr_dsk' in locals() and globals():
        pass
    else:
        fr_dsk = defdisk
    if 'fr_sq' in locals() and globals():
        pass
    else:
        fr_sq = 1
    if 'nmaps' in locals() and globals():
        pass
    else:
        nmaps = 1
    if 'flux' in locals() and globals():
        pass
    else:
        flux = {'': [0, 0, 0, 0]}
    if 'niter' in locals() and globals():
        pass
    else:
        niter = 100
    if 'grid_flag' in locals() and globals():
        pass
    else:
        grid_flag = 0
    if 'gridsource' in locals() and globals():
        pass
    else:
        gridsource = ''
    if 'n_grid' in locals() and globals():
        pass
    else:
        n_grid = 0
    if 'm_grid' in locals() and globals():
        pass
    else:
        m_grid = 0
    if 'grid_offset' in locals() and globals():
        pass
    else:
        grid_offset = 0
    if 'dual_geo' in locals() and globals():
        pass
    else:
        dual_geo = 0
    if 'arch_user' in locals() and globals():
        pass
    else:
        arch_user = ''
    if 'arch_pass' in locals() and globals():
        pass
    else:
        arch_pass = ''
    if 'file' in locals() and globals():
        pass
    else:
        file = []
    if 'kntr_flag' in locals() and globals():
        pass
    else:
        kntr_flag = 0
    if 'fittp_flag' in locals() and globals():
        pass
    else:
        fittp_flag = 0
    if 'get_key_flag' in locals() and globals():
        pass
    else:
        get_key_flag = 0
    if 'code' in locals() and globals():
        pass
    else:
        code = ''
    if 'max_ant' in locals() and globals():
        pass
    else:
        max_ant = 12
    if 'phase_cal_flag' in locals() and globals():
        pass
    else:
        phase_cal_flag = 0
    if 'amp_cal_flag' in locals() and globals():
        pass
    else:
        amp_cal_flag = 0
    if 'imna' in locals() and globals():
        pass
    else:
        imna = ''
    if 'phase_target_flag' in locals() and globals():
        pass
    else:
        phase_target_flag = ''
    if 'amp_target_flag' in locals() and globals():
        pass
    else:
        amp_target_flag = ''
    if 'antennas' in locals() and globals():
        pass
    else:
        antennas = [0]
    if 'refeed_flag' in locals() and globals():
        pass
    else:
        refeed_flag = 0
    if 'plot_tables' in locals() and globals():
        pass
    else:
        plot_tables = -1
    if 'dofit' in locals() and globals():
        pass
    else:
        dofit = [0]
    if 'apply_selfcal' in locals() and globals():
        pass
    else:
        apply_selfcal = 0
    if 'tysmo_flag' in locals() and globals():
        pass
    else:
        tysmo_flag = 0
    if 'solint' in locals() and globals():
        pass
    else:
        solint = 0
    if 'smodel' in locals() and globals():
        pass
    else:
        smodel = [1, 0]
    if 'uvwtfn' in locals() and globals():
        pass
    else:
        uvwtfn = ''
    if 'robust' in locals() and globals():
        pass
    else:
        robust = 0
    if 'bandcal' in locals() and globals():
        pass
    else:
        bandcal = ['']
    if 'do_band_flag' in locals() and globals():
        pass
    else:
        do_band_flag = 0
    if 'dpfour' in locals() and globals():
        pass
    else:
        dpfour = 0
    if 'min_elv' in locals() and globals():
        pass
    else:
        min_elv = 0
    if 'rpossm_flag' in locals() and globals():
        pass
    else:
        rpossm_flag = 0
    if 'ma_sad_flag' in locals() and globals():
        pass
    else:
        ma_sad_flag = 0
    if 'plot_map' in locals() and globals():
        pass
    else:
        plot_map = 0
    if 'min_snr' in locals() and globals():
        pass
    else:
        min_snr = 7
    if 'smooth' in locals() and globals():
        pass
    else:
        smooth = [0]
    if 'beam' in locals() and globals():
        pass
    else:
        beam = [0, 0, 0]
    if 'TECU_model' in locals() and globals():
        pass
    else:
        TECU_model = 'jplg'

    ##############################################################################
    # Start main script

    logging.info('######################')
    logging.info('%s', get_time())
    logging.info('###################### ')

    # constrain time range for fringe finder?
    #mp_timera = [0, 0, 0, 0, 0, 0, 0, 0]
    #TODO
    mp_timera = [0, 23, 00, 1, 0, 23, 2, 59]
    bandcal = p_ref_cal # Bandpass calibrator


    #################
    # Split Options #
    #################

    smooth = [0, 0, 0]  # Smooth during split for line data
    split_outcl = 'SPLIT'  # outclass in SPLIT '' => 'SPLIT'

    ##################################
    # Optional inputs for fringe fit #
    ##################################

    [fr_n, fr_c, fr_d, fr_s] = ['', '', 1, 1]
    # Input image to use in FRINGE:
    # ['NAME','CLASS',DISK,SEQUENCE]
    smodel = [1, 0]  # SMODEL in FRING
    solint = 0  # SOLINT in FRING
    nmaps = 1  # NMAPS in FRING

    logging.info("FILE NAME %s", filename[0])
    logging.info("OUT  NAME %s ", outname[0])
    # print("FILE NAME =========", filename[0])  # ,filename[1],filename[2])
    # print("OUT  NAME =========", outname[0])  # ,outname[1],outname[2])
    if not os.path.exists(outname[0]):
        os.mkdir(outname[0])
    ##################################
    # Data preparation and first prep#
    ##################################

    load_flag = 0  # Load data from disk?
    listr_flag = 0  # Print out LISTR?
    dtsum_flag = 0  # Run dtsum to check antena participation?
    tasav_flag = 0  # Run tasav on original tables?
    geo_prep_flag = 0  # Run TECOR and EOP corrections? and uvflg for evn data
    # get_key_flag    = 0        # Download key-file from archive
    # RDBE_check      = 0        # Check Geoblock data for RDBE errors?
    if step1 == 1:
        load_flag = 1
        listr_flag = 1
        tasav_flag = 1
        geo_prep_flag = 3
        dtsum_flag = 1

    if load_flag == 1:
        loadindx(file_path, filename[0], outname[0], outclass[0], outdisk[0],
                 nfiles[0], ncount[0], doconcat[0], antname, logfile+'-test')
        # for i in range(n):
        #    loadindx(file_path, filename[i], outname[i], outclass[i], outdisk[i], nfiles[i], ncount[i], doconcat[i], antname, logfile)

    data = range(n)

    logging.info('################################## ')
    for i in range(n):
        data[i] = AIPSUVData(outname[i], outclass[i], int(outdisk[i]), int(1))
        if data[i].exists():
            data[i].clrstat()
        if dtsum_flag == 1:
            rundtsum(data[i])
        if listr_flag == 1:
            runlistr(data[i])
    logging.info('################################## ')

    # Download TEC maps and EOPs

    if pr_prep_flag == 1 or geo_prep_flag >= 1:
        (year, month, day) = get_observation_year_month_day(data[0])
        num_days = get_num_days(data[0])

        doy = get_day_of_year(year, month, day)

        get_TEC(year, doy, TECU_model, geo_path)
        get_eop(geo_path)

        if num_days == 2:
            get_TEC(year, doy + 1, TECU_model, geo_path)

    logging.info('################################## ')
    logging.info(get_time())
    logging.info('################################## ')

    if geo_prep_flag > 0:
        geo_data = data[geo_data_nr]
        # runuvflg(geo_data,flagfile[geo_data_nr],logfile)
        check_sncl(geo_data, 0, 1, logfile)
        if geo_data.header['telescop'] == 'EVN':
            if geo_prep_flag == 1:
                runTECOR(geo_data, year, doy, num_days, 3, TECU_model)
            else:
                runtacop(geo_data, geo_data, 'CL', 1, 3, 0)
        else:
            #TODO ADD GEO---
            if geo_prep_flag == 1:
                runTECOR(geo_data, year, doy, num_days, 2, TECU_model)
                runeops(geo_data, geo_path)
            elif geo_prep_flag == 2:#no tecr
                runtacop(geo_data, geo_data, 'CL', 1, 2, 0)
                runeops(geo_data, geo_path)
            elif geo_prep_flag == 3:#no eop
                runTECOR(geo_data, year, doy, num_days, 3, TECU_model)

    pr_data = data[0]

    if tasav_flag == 1:
        if flagfile[0] != '':
            runuvflg(pr_data, flagfile[0], logfile)
        if antabfile[0] != '':
            runantab(pr_data, antabfile[0])
        runtasav(pr_data, 0, logfile)

        # todo : possom choose time range
        # pass
    ###################################################################
    # Data inspect    

    if inspect_flag == 1:
        timerange,N_obs=get_fringe_time_range(data[0],calsource[0])
        N_ant,refants=get_refantList(data[0])
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>")
        print (timerange,N_ant,refants,N_obs)
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>")
        possmplot(data[0],sources='',timer=timerange,gainuse=3,flagver=0,stokes='HALF',nplot=9,bpv=0,ant_use=[0],cr=1)
        possmplot(data[0],sources='',timer=timerange,gainuse=3,flagver=0,stokes='HALF',nplot=2,bpv=0,ant_use=[0],cr=0)
        if antname == 'VLBA':
            runsnplt(data[0],inver=1,inex='TY',sources='',optype='TSYS',nplot=4,timer=[])
    if inspect_flag ==2:
        if antname != 'VLBA':
            runsnplt(data[0],inver=1,inex='TY',sources='',optype='TSYS',nplot=4,timer=[])
        possmplot(data[0],sources=p_ref_cal[0],timer=RFIck_tran,gainuse=3,flagver=0,stokes='HALF',nplot=2,bpv=0,ant_use=[0],cr=0)
        print (data[0].sources)
    if inspect_flag == 3:
        timerange,N_obs=get_fringe_time_range(data[0],calsource[0])
        N_ant,refants=get_refantList(data[0])
        refant      = refants[0]
        refant_candi= refants[1:]+[0]
        if os.path.exists(parms_filename):
            os.remove(parms_filename)
        sys.stdout = open(parms_filename,'w')
        print (N_ant,N_obs)
        print (timerange)
        print (refant)
        print (refant_candi)
        sys.stdout = sys.__stdout__

    logging.info('############################')
    logging.info('Data inspection before apcal')
    logging.info('############################')


    #=============================================================================

    lines=open(parms_filename,'r').read()
    lines=lines.splitlines()

    refant      = int(lines[2])          # refant=0 to select refant automatically
    refant_candi=[]
    possm_scan =[]
    b = lines[3]     # candidate refant for search in fringe
    c = lines[1]
    for i in b.split(','):
        refant_candi.append(int(i.strip().strip('[]')))
    for i in c.split(','):
        possm_scan.append(int(i.strip().strip('[]')))

    print (possm_scan)
    apfive      = 1		#This sets the aparm(5) in fringe (how to combine IFs in the global finge fitting: 0=not combine, 1=combine all, 3=combine in halves
                        # phase ref calbrator sources '' => automatically
    split_seq   = 1		#for muti pyfiles in the same username, set this differently will avoid bugs during split and fittp, if not, use 1 only.
    targets     = target + p_ref_cal
    mp_source   = calsource             # fringe finder     '' => automatically
    mp_timera   = possm_scan             # constrain time range for fringe finder?
    bandcal     = calsource     # Bandpass calibrator
    flagver     = 1  		     # Flag table version to be used
    tyver       = 1                      # Tsys table version to be used
    chk_trange  = [0]                    #timerange on p_cal for possm checking
    # 1 for all VLGA, 0/-1 not do
    dofit=-1
    #todo 
    #dofit=  [-1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,-1]  #usually for EVN with tsys antenas   

    #########################################################################
    # second-run--data_calibration #
    #########################################################################
    #step2           = 0	   # Auto control of the seond block
    apcal_flag      = 0        # Do amplitude calibration?
    pang_flag       = 0        # Run PANG?
    pr_fringe_flag  = 0        # Do manual phase cal?
    do_fringe_flag  = 0        # Do first run of fringe cal on all sources?
    plot_first_run  = 0        # DO possm and snplt to check first run result?
    do_band_flag    = 0
    split_1_flag    = 0        # Split calibrated data in first run?
    if step2 >= 1:# if step2 = 1: normal p-ref, if step2=2, do self-cal for targets. Do not set step2 >=3
        apcal_flag      = 2-step2
        pang_flag       = 2-step2
        pr_fringe_flag  = 2-step2
        do_fringe_flag  = step2
        plot_first_run  = 2-step2
        do_band_flag    = 1
        split_1_flag    = step2
    #todo step2
    if step2 == 1:
        if apcal_flag == 1:
            check_sncl(pr_data, 0, 3, logfile)
            # if antabfile[i]!='':
            #   runantab(pr_data,antabfile[i])
            if tysmo_flag == 1:
                runtysmo(pr_data, 90, 10)
            print pr_data.header['telescop']
            if antname == 'EVN':
                runapcal(pr_data, tyver, 1, 1, dofit, 'GRID')
                runclcal(pr_data, 1, 3, 4, '', 1, refant)
                runtacop(pr_data, pr_data, 'SN', 1, 2, 1)
                runtacop(pr_data, pr_data, 'CL', 4, 5, 1)
            elif antname == 'VLBA':
                runaccor(pr_data)
                runclcal(pr_data, 1, 3, 4, 'self', 1, refant)
                runapcal(pr_data, tyver, 1, 2, 1, 'GRID')
                runclcal(pr_data, 2, 4, 5, '', 1, refant)
            elif antname == 'LBA' :  # for LBA
                runaccor(pr_data)
                runclcal(pr_data, 1, 3, 4, 'self', 1, refant)
                runapcal(pr_data, tyver, 1, 2, -1, 'GRID')
                runclcal(pr_data, 2, 4, 5, '', 1, refant)
            else:
                print("Error ANT : choose EVN/VLBA/LBA")

            logging.info('####################################')
            logging.info(get_time())
            logging.info('####################################')

        if pang_flag == 1:
            check_sncl(pr_data, 2, 5, logfile)
            runpang2(pr_data)
            logging.info('####################################')
            logging.info('Finish PANG')
            logging.info('####################################')

        so = ''
        ti = ''
        if pr_fringe_flag == 1 and step2 == 1:
            logging.info('####################################')
            logging.info('Begin mannual phase-cal')
            logging.info('####################################')
            check_sncl(pr_data, 2, 6, logfile)
            # if refant_flag==1:
            #    refant=select_refant2(pr_data, logfile)
            so, ti = man_pcal(pr_data, refant, mp_source, mp_timera, debug, logfile, dpfour)
            print(so, ti)
        if n == 1:
            so_ti = [so, ti]
        if n == 2:
            if so_ti[0] == so and so_ti[1] == ti:
                logging.info('#############################################')
                logging.info( '### Both manual phasecal scans identical. ###' )
                logging.info('#############################################')
            else:
                logging.info('#############################################')
                logging.info( '### Manual phasecal scans different.      ###' )
                logging.info('### Select one manually.                  ###')
                logging.info('#############################################')
                sys.exit()
            # runclcal(pr_data, 3, 6, 7, '', 0, refant,[0], [''])
        runclcal2(pr_data, 3, 6, 7, '2pt', 0, refant, [0], mp_source, '')
        if do_fringe_flag == 1:
            logging.info('####################################')
            logging.info('Begin first fringe')
            logging.info('####################################')
            check_sncl(pr_data, 3, 7, logfile)
            fringecal_ini(pr_data, refant, refant_candi, calsource[0], 7, 1, solint, -1, 0)
            fringecal_ini(pr_data, refant, refant_candi, p_ref_cal[0], 7, 1, solint, -1, 0)
            # fringecal_ini(pr_data,refant, refant_candi, p_ref_cal,7,1,solint,-1,0)
            runclcal2(pr_data, 4, 7, 8, 'ambg', -1, refant, [0], calsource, calsource)
            runclcal2(pr_data, 5, 7, 9, 'ambg', 1, refant, [0], p_ref_cal[0], targets)
        if do_fringe_flag == 2:
            logging.info('####################################')
            logging.info('Begin first fringe')
            logging.info('####################################')
            check_sncl(pr_data, 3, 7, logfile)
            fringecal_ini(pr_data, refant, refant_candi, calsource[0], 7, 1, solint, -1, 0)
            fringecal_ini(pr_data, refant, refant_candi, targets, 7, 1, solint, -1, 0)
            runclcal2(pr_data, 4, 7, 8, 'ambg', -1, refant, [0], calsource, calsource)
            runclcal2(pr_data, 5, 7, 9, 'ambg', -1, refant, [0], targets, targets)
            # fringecal_ini(indata, refant, refant_candi, calsource, gainuse, flagver, solint, doband, bpver)
        if plot_first_run == 1:
            # check_sncl(pr_data,5,7,logfile)
            runsnplt(pr_data, inver=9, inex='CL', sources=targets, optype='PHAS', nplot=4, timer=[])
            runsnplt(pr_data, inver=5, inex='SN', sources=targets, optype='PHAS', nplot=4, timer=[])
            runsnplt(pr_data, inver=5, inex='SN', sources=targets, optype='DELA', nplot=4, timer=[])
            runsnplt(pr_data, inver=5, inex='SN', sources=targets, optype='RATE', nplot=4, timer=[])
            possmplot(pr_data, sources=p_ref_cal[0], timer=chk_trange, gainuse=9, flagver=flagver, stokes='HALF', nplot=9, bpv=0,
                    ant_use=[0])
        logging.info('####################################')
        logging.info(get_time())
        logging.info('####################################')
        if do_band_flag == 1:
            check_sncl(pr_data, 5, 9, logfile)
        if pr_data.table_highver('AIPS BP') >= 1:
            pr_data.zap_table('AIPS BP', -1)
            do_band(pr_data, bandcal, 8, 1, logfile)
        else:
            do_band(pr_data, bandcal, 8, 1, logfile)
            possmplot(pr_data, sources=p_ref_cal[0], timer=chk_trange, gainuse=9, flagver=0, stokes='HALF', nplot=9, bpv=1,
                    ant_use=[0])
            possmplot(pr_data, sources=bandcal[0], timer=possm_scan, gainuse=8, flagver=0, stokes='HALF', nplot=9, bpv=1,
                    ant_use=[0])

        #line_data = data[line]
        #cont_data = data[cont]
        line_data = data[0]
        cont_data = data[0]
        line_data2 = AIPSUVData(line_data.name, line_data.klass, line_data.disk, 2)
        cont_data2 = AIPSUVData(cont_data.name, cont_data.klass, cont_data.disk, 2)

        if bandcal == ['']:
            doband = -1
            bpver = -1
        else:
            doband = 1
            bpver = 1

        if split_1_flag == 1:
            check_sncl(cont_data, 5, 9, logfile)
            run_split2(cont_data, calsource[0], 8, split_outcl, doband, bpver, flagver, split_seq)
            run_split2(cont_data, p_ref_cal[0], 9, split_outcl, doband, bpver, flagver,split_seq)
            #todo for multi phase cal
            if len(p_ref_cal)>=2:
                run_split2(cont_data, p_ref_cal[1], 9, split_outcl, doband, bpver, flagver,split_seq)
                #run_fittp_data(source, split_outcl, defdisk, logfile)
                run_split2(cont_data, target[0], 9, split_outcl, doband, bpver, flagver,split_seq)

    ##################################
    # Optional inputs for fringe fit #
    ##################################
    #Step3
    #TODO add run_difmap.py
    #os.system('python3 run_difmap.py')
    # TODO
    fr_path='/data/VLBI/code/vlbi-pipeline/vlbi-pipeline/BB203A/'
    # TODO Phase cal
    #fr_file=p_ref_cal'J1339+6328_SPLIT_1-cln.fits'
    fr_file='J1339+6328_SPLIT_1-cln.fits'
    #TODO fr_nm limit to 6 chars
    [fr_nm, fr_cls, fr_dsk, fr_sq] = [fr_file[0:4]+ '-fr','CLN',1,1]
                                                # Input image to use in FRINGE:
                                                # ['NAME','CLASS',DISK,SEQUENCE]
    smodel = [0,0]                 # SMODEL in FRING                                
    solint = 4                     # SOLINT in FRING
    nmaps  = 1                     # NMAPS  in FRING
    nofit_rate = 1     #if =1, suppress rate in fringe(dparm8), if=0 fit rate as usual
    #ftrate=1
    cluse=7
    sncor_timer =[0]
    if step3 == 1:
        ld_fr_fringe_flag  = 1
        do_fr_fringe_flag  = 1
        #do_calib_1_flag       = 1
        check_delay_rate = 1
        split_2_flag     = 1

    ##zyk+++
    if ld_fr_fringe_flag == 1:
        fr_image = AIPSImage(fr_nm, fr_cls, fr_dsk, fr_sq)
        if fr_image.exists():
            pass
        else:
            loadfr(fr_path, fr_file, fr_nm, fr_cls, fr_dsk, antname, logfile)

    if do_fr_fringe_flag == 1:
        cont_data = data[0]
        check_sncl(cont_data, 3, 7, logfile)
        doband = 0
        bpver = -1
        fringecal(cont_data, fr_image, nmaps, 7, refant, refant_candi, p_ref_cal[0], solint, smodel, doband, bpver, dpfour, logfile)
        runclcal2(cont_data, 4, 7, 8, 'AMBG', -1, refant, [0], p_ref_cal[0], targets)
    if check_delay_rate == 1:
        # chk_sn_cl(cont_data,6,10,p_ref_cal[0],chk_trange,1)
        #chk_sn_cl(cont_data, 5, 9, p_ref_cal[0], chk_trange, 1)
        chk_sn_cl(cont_data, 4, 8, p_ref_cal[0], chk_trange, 1, flagver)
        runsnplt(pr_data, inver=8, inex='CL', sources=targets, optype='PHAS', nplot=4, timer=[])
        runsnplt(pr_data, inver=4, inex='SN', sources=targets, optype='PHAS', nplot=4, timer=[])
        runsnplt(pr_data, inver=4, inex='SN', sources=targets, optype='DELA', nplot=4, timer=[])
        runsnplt(pr_data, inver=4, inex='SN', sources=targets, optype='RATE', nplot=4, timer=[])
        # runsnplt(pr_data,inver=7,inex='SN',sources='',optype='DELA',nplot=4,timer=[])
        # runsnplt(pr_data,inver=7,inex='SN',sources='',optype='RATE',nplot=4,timer=[])

    if split_2_flag >= 1:
        #check_sncl(cont_data, 5, 9, logfile)
        check_sncl(cont_data, 4,8, logfile)
        # run_split2(cont_data, p_ref_cal[0], 10, 'SCL10', doband, bpver, flagver)
        run_split2(cont_data, target[0], 8, 'SCL10', doband, bpver, flagver, 1)
        # run_split2(cont_data, p_ref_cal[0], 11, 'SCL11', doband, bpver, flagver)
        if split_2_flag >= 2:
            run_split2(cont_data, p_ref_cal[0], 8, 'SCL10', doband, bpver, flagver, 1)


if __name__ == '__main__':
    # current_time()
    logfilename = 'logs/vlbi-pipeline.' + current_time() + '.log'
    run_main(logfilename)
