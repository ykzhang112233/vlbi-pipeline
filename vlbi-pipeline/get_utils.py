#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
from utils import *
from config import *
import logging

# Get the best scan and good ref antenna (from Sumit)

def isinde(number):
    INDE = 3140.892822265625
    return abs(number - INDE) < 1e-12

def get_fringe_time_range(uvdata, fringe_cal):
    '''
    Get time range coverd by a scan of a bright calibrator to be used
    for short duration fringe fitting
    '''
    timerange = [0, 0, 0, 0, 0, 0, 0, 0]
    # First check the existence of fringe calibrator in the list of scans:
    SUtab = uvdata.table('SU', 1)
    for row in SUtab:
        # print row
        # print fringecal
        source = row.source.strip()
        if source == fringe_cal:
            fringeCal_id = row.id__no
            fringeCal_SUexist = True
    try:
        fringeCal_SUexist
    except NameError:
        fringeCal_SUexist = False
    if fringeCal_SUexist == False:
        logging.info('The fringe fitter is not in the SU table!')
        sys.exit()
    NXtab = uvdata.table('NX', 1)
    for row in NXtab:
        # print row
        if fringeCal_id == row.source_id:
            fringeCal_NXexist = True
    try:
        fringeCal_NXexist
    except NameError:
        fringeCal_NXexist = False
    if fringeCal_NXexist == False:
        logging.info("The fringe finder ", fringe_cal,
                     " is not in the NX table!")
        logging.warning("The fringe finder ", fringe_cal,
                        " is not in the NX table!")
        sys.exit()
    # Finding the number of not-completely flagged antennas:
    N_ant, refantList = get_refantList(uvdata)
    # If there is no antenna which is completely flagged, N_ant=len(uvdata.antennas)
    # Running task BSCAN to get start and end times (in days from the reference date) for which all the antennas are present:
    bscan = AIPSTask('BSCAN')
    bscan.default()
    bscan.indata = uvdata
    bscan.sources[1:] = [fringe_cal]
    bscan.go()
    # print help(bscan)
    # print bscan.outputs()
    delete_temp()
    # Printing the console output to a text file:
    sys.stdout = open('tmp_test*.txt', 'w')
    bscan.outputs()
    sys.stdout = sys.__stdout__
    lines = open('tmp_test*.txt', 'r').readlines()
    with open('tmp_test1.txt', 'w') as outfile:
        line = lines[0]
        line = line.split(': ')[1]
        outfile.write(line)
    # Reading list of lists from a text file:
    lines = open("tmp_test1.txt", "r").read()
    list_of_lists = eval(lines)
    print(list_of_lists)

    list_of_lists1 = []
    N_obs = []
    for list_i in list_of_lists:
        print(list_i)
        if list_i != [0.0, 0.0, 0.0]:
            if int(list_i[2]) == N_ant:
                list_of_lists1.append(list_i)
            else:
                list_of_lists1.append(list_i)
            #todo get the first value
            N_obs.append(list_i[2])
    # print list_of_lists1
    if len(list_of_lists1) == 0:
        logging.info("There is no best scan of fringe fitter ",
                     fringe_cal,  ". Try to change the fringe fitter!")
        sys.exit()
    #todo add later
    #else:
    #    logging.info("There are %d best scans of fringe fitter '%s' having %d antennas present.".format( len(list_of_lists1), fringe_cal, N_obs[0]))
    print ("################")
    print (list_of_lists)
    print (list_of_lists1)
    print (fringe_cal)
    print (N_obs)
    print ("################")

    #delete_temp()

    # Finding start and end time of the mid scan of the fringe fitter (in case of many scans) for which all antennas are present:
    if len(list_of_lists1) % 2 == 1:
        # median value for odd number of scans of fringe fitting calibrator.
        x = int(len(list_of_lists1) / 2)
    else:
        x = int((len(list_of_lists1) - 1) / 2)  # for even number of scans
    print(x)
    startTime0 = list_of_lists1[x][0]  # in days
    endTime0 = list_of_lists1[x][1]  # in days
    # Estimating fringe-fit time range for the selected scan of fringe-fitter:
    # solint=get_solint(uvdata)
    solint = 1  # in minutes
    scanLength = (endTime0 - startTime0) * 24 * 60  # in minutes
    if scanLength <= solint:
        startTime = startTime0
        endTime = endTime0
    elif scanLength < solint + (10 / 60):
        endTime = endTime0 - 1.0 / (24 * 60 * 60)
        startTime = endTime - (solint + 0.001) / (24 * 60)
    else:
        # ignoring 5 seconds of data from end of the scan.
        endTime = endTime0 - 5.0 / (24 * 60 * 60)
        startTime = endTime - (solint + 0.001) / (24 * 60)
    timerange[0] = int(np.floor(startTime))
    startTime1 = 24 * (startTime - timerange[0])
    timerange[1] = int(np.floor(startTime1))
    startTime2 = 60 * (startTime1 - timerange[1])
    timerange[2] = int(np.floor(startTime2))
    timerange[3] = int(np.ceil(60 * (startTime2 - timerange[2])))
    timerange[4] = int(np.floor(endTime))
    endTime1 = 24 * (endTime - timerange[4])
    timerange[5] = int(np.floor(endTime1))
    endTime2 = 60 * (endTime1 - timerange[5])
    timerange[6] = int(np.floor(endTime2))
    timerange[7] = int(np.floor(60 * (endTime2 - timerange[6])))
    return timerange, N_obs


def get_central_antennas(uvdata):
    '''
    Get the list of central antennas in the array

    Finding the list of central antennas' number given in ascending order of distance from UV center:
    '''
    ANtab = uvdata.table('AN', 1)
    Sum_baselineLength_3d = {}
    for i, row in enumerate(ANtab):
        # print row
        Sum_baselineLength_3d["{}".format(i)] = []
        for row2 in ANtab:
            xsep = row.stabxyz[0] - row2.stabxyz[0]
            ysep = row.stabxyz[1] - row2.stabxyz[1]
            zsep = row.stabxyz[2] - row2.stabxyz[2]
            baselineLength_3d = np.sqrt(
                (xsep * xsep) + (ysep * ysep) + (zsep * zsep))
            Sum_baselineLength_3d["{}".format(i)].append(baselineLength_3d)
        Sum_baselineLength_3d["{}".format(i)] = np.sum(
            Sum_baselineLength_3d["{}".format(i)])
    Sum_baselineLength_3d = pd.DataFrame.from_dict(
        Sum_baselineLength_3d, orient='index')
    Sum_baselineLength_3d.reset_index(drop=False, inplace=True)
    Sum_baselineLength_3d.rename(
        columns={'index': 'ant_num', 0: 'sumBaselineLength'}, inplace=True)
    Sum_baselineLength_3d['ant_num'] = Sum_baselineLength_3d['ant_num'].astype(
        int) + 1
    Sum_baselineLength_3d.sort_values(
        by=['sumBaselineLength'], ascending=True, inplace=True)
    centralAntennas = list(Sum_baselineLength_3d['ant_num'])
    return centralAntennas


def get_refantList(uvdata):
    '''
    Get the 3 reference antenna list to be used in global fringe-fitting:
    Finding the list of central antennas' number given in ascending order of distance from UV center:
    '''
    centralAntennas = get_central_antennas(uvdata)
    # Removing temp files:
    delete_temp()
    # Running DTSUM task to get visibility data summary for each antenna:
    dtsum = AIPSTask('DTSUM')
    dtsum.default()
    dtsum.indata = uvdata
    dtsum.docrt = -3  # To suppress header information while writing.
    dtsum.outprint = 'tmp_test*.txt'
    dtsum.go()
    # Modifying the visibility data summary text file:
    with open('tmp_test*.txt', 'r') as infile:
        with open('tmp_test1.txt', 'w') as outfile:
            prtline = False
            for line in infile:
                if '---' in line:
                    prtline = True
                if prtline:
                    outfile.write(line)
    lines = open('tmp_test1.txt', 'r').readlines()
    # open('tmp_test2.txt', 'w').writelines(lines[1:-1])
    with open('tmp_test2.txt', 'w') as outfile:
        for line in lines[1:-1]:
            # line=line.split('|')[1]
            line = line.replace(' | ', ' ', 1)
            outfile.write(line)
    # Read ASCII table file as a dataframe:
    df1 = pd.read_csv('tmp_test2.txt', sep='\s+', na_filter=False, header=None)
    # print df1
    # Making a column as the dataframe row indices:
    df2 = df1.set_index(keys=df1[0], drop=True, append=False, inplace=False)
    df2.drop(df2.columns[[0]], axis=1, inplace=True)
    # print df2
    # Modifying the dataframe element-wise:
    for i in range(len(df2)):
        for j in range(len(df2)):
            if df2.iloc[i, j] == 0:
                df2.iloc[i, j] = df2.iloc[j, i]
    # print df2
    # Averaging each antenna column's visibilities:
    antMeanVisibilities = df2.mean(axis=0).values
    # Taking care of any completely flagged antenna:
    df3 = pd.DataFrame()
    df3[0] = df1[0]
    df3[1] = antMeanVisibilities
    # print df3
    # Number of available antennas:
    N_ant = len(df3)
    # Removing temp files:
    delete_temp()
    # Find the 3 central antennas with maximum number of visibilities out of first 5 central antennas:
    x = []
    y = []
    for i in range(5):
        j = centralAntennas[i]
        k = 0.0
        for a in range(len(df3)):
            if df3.iloc[a, 0] == j:
                k = df3.iloc[a, 1]
        if k == 0.0:
            j = np.nan
        x.append(j)
        y.append(k)
    x = np.array(x)
    x1 = x[~np.isnan(x)]
    refantList = x1[:3].tolist()
    # print refantList
    print("N_ant ")
    print(N_ant)
    print(refantList)
    import time
    return N_ant, refantList

##############################################################################
# Download TEC maps
#


def get_TEC(year, doy, TECU_model, geo_path):
    year = str(year)[2:4]
    if doy < 10:
        doy = '00' + str(doy)
    elif doy < 100:
        doy = '0' + str(doy)
    else:
        doy = str(doy)

    name = TECU_model + doy + '0.' + year + 'i'
    #    name4='esag'+doy+'0.'+year+'i'
    #    if os.path.exists(name) or os.path.exists(name2):
    if os.path.exists(geo_path + name):
        print
        'TEC File already there.'
    else:
        path = 'https://cddis.nasa.gov/archive/gps/products/ionex/20' + year + '/' + doy + '/'
        # todo make sure .netrc exists
        os.popen(
            r'curl -c cookies.curl --netrc-file ~/.netrc -n -L -O ' + path + name + '.Z')
        # todo make sure the file download
        os.popen(r'uncompress -f ' + name + '.Z')
        os.popen(r'mv ' + name + ' ' + geo_path)


'''
CDDIS Archive Access: .netrc instructions
In order for cURL to use those credentials you will need to create a .netrc file.

To create a .netrc file, you will need to create a text file with the name .netrc; this file needs to have read permissions set to only yourself, so that no one can read your file and get your username and password. The format of the file is:

machine urs.earthdata.nasa.gov login <username> password <password>

where <username> and <password> are the values you set when you created your Earthdata login account.
'''


##############################################################################
# Download EOP file
#
def get_eop(geo_path):
    os.popen( r'curl -c cookies.curl --netrc-file ~/.netrc -n -L -O "https://cddis.nasa.gov/archive/vlbi/gsfc/ancillary/solve_apriori/usno_finals.erp"')
    os.popen(r'mv usno_finals.erp ' + geo_path)
    '''
    if os.path.exists(geo_path + 'usno_finals.erp'):
        os.popen(
            r'curl -c cookies.curl --netrc-file ~/.netrc -n -L -O "https://cddis.nasa.gov/archive/vlbi/gsfc/ancillary/solve_apriori/usno_finals.erp"')
        # age = (time.time() - os.stat(eop_path+'usno_finals.erp')[8])/3600
        # if age<12: pass
        # else:
        #    os.popen(r'wget http://gemini.gsfc.nasa.gov/solve_save/usno_finals.erp')
        #    os.popen(r' rm -rf '+eop_path+'usno_finals.erp')
        #    os.popen(r'mv usno_finals.erp '+eop_path)
        print ('---> Use downloaed erp file')
        # --- ZB
    else:
        # todo make sure .netrc exists
        os.popen(
            r'curl -c cookies.curl --netrc-file ~/.netrc -n -L -O "https://cddis.nasa.gov/archive/vlbi/gsfc/ancillary/solve_apriori/usno_finals.erp"')
        os.popen(r'mv usno_finals.erp ' + geo_path)
    '''

def get_time():
    t = range(6)
    t[0] = time.localtime()[0]
    t[0] = str(t[0])
    for i in range(1, 6):
        t[i] = time.localtime()[i]
        if t[i] < 10:
            t[i] = '0'+str(t[i])
        else:
            t[i] = str(t[i])
    a = t[3]+':'+t[4]+':'+t[5]+' on '+t[0]+'/'+t[1]+'/'+t[2]
    return a


def get_observation_year_month_day(aips_data):
    '''
    Get the day ot year/month/day for the start of observation
    '''
    date_string = aips_data.header.date_obs
    date_list = date_string.split('-')
    year = int(date_list[0])
    month = int(date_list[1])
    day = int(date_list[2])
    return (year, month, day)


def get_num_days(indata):
    '''
    Get number of days
    '''
    nx_table = indata.table('AIPS NX', 0)
    n = len(nx_table)
    num_days = int(nx_table[n - 1]['time'] + 1)
    return num_days


def get_day_of_year(year, month, day):
    '''
    Get the doy from year/month/day
    '''
    day_of_year_list = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    doy = day_of_year_list[month - 1] + day
    if (month > 2):
        if ((year & 0x3) == 0):
            if ((year % 100 != 0) or (year % 400 == 0)):
                doy = doy + 1
    return doy


def get_ant(data):
    antennas = {}
    for row in data.table('AN', 0):
        antennas[row.nosta] = row.anname[0:2]
    return antennas
##############################################################################
# Find best scan for manual phasecal
#
def get_best_scan(indata, logfile, qualfile, do_write):
    sn_table = indata.table('AIPS SN', 0)
    naxis = indata.header['naxis']
    sources = get_sources(indata)

    t = [0]
    qf = [0]
    snr = []
    tr = []
    sid = []

    n = 0
    max_sol = 0

    sid.append(sn_table[0]['source_id'])
    tr.append(get_timerange_tab(indata, 'AIPS SN', 0))
    snr.append([])

    if naxis[3] > 1:
        for j in range(naxis[3]):
            if isinde(sn_table[0]['delay_1'][j]) == False:
                t[n] += 1
                qf[n] = qf[n] + 1. / sn_table[0]['weight_1'][j] ** 2
                snr[n].append(sn_table[0]['weight_1'][j])

        for i in range(1, len(sn_table)):
            if sn_table[i]['time'] == sn_table[i - 1]['time']:
                for j in range(naxis[3]):
                    if isinde(sn_table[i]['delay_1'][j]) == False:
                        t[n] += 1
                        qf[n] = qf[n] + 1. / sn_table[i]['weight_1'][j] ** 2
                        snr[n].append(sn_table[i]['weight_1'][j])
                if t[n] > max_sol:
                    max_sol = t[n]
                    id = n
            else:
                t.append(0)
                qf.append(0)
                snr.append([])
                n += 1
                sid.append(sn_table[i]['source_id'])
                tr.append(get_timerange_tab(indata, 'AIPS SN', i))
                for j in range(naxis[3]):
                    if isinde(sn_table[i]['delay_1'][j]) == False:
                        t[n] += 1
                        qf[n] = qf[n] + 1. / sn_table[i]['weight_1'][j] ** 2
                        snr[n].append(sn_table[i]['weight_1'][j])
                if t[n] > max_sol:
                    max_sol = t[n]
                    id = n

    elif naxis[3] == 1:
        if isinde(sn_table[0]['delay_1']) == False:
            t[n] += 1
            qf[n] = qf[n] + 1. / sn_table[0]['weight_1'] ** 2
            snr[n].append(sn_table[0]['weight_1'])

        for i in range(1, len(sn_table)):
            if sn_table[i]['time'] == sn_table[i - 1]['time']:
                if isinde(sn_table[i]['delay_1']) == False:
                    t[n] += 1
                    qf[n] = qf[n] + 1. / sn_table[i]['weight_1'] ** 2
                    snr[n].append(sn_table[i]['weight_1'])
                if t[n] > max_sol:
                    max_sol = t[n]
                    id = n
            else:
                t.append(0)
                qf.append(0)
                snr.append([])
                n += 1
                sid.append(sn_table[i]['source_id'])
                tr.append(get_timerange_tab(indata, 'AIPS SN', i))
                if isinde(sn_table[i]['delay_1']) == False:
                    t[n] += 1
                    qf[n] = qf[n] + 1. / sn_table[i]['weight_1'] ** 2
                    snr[n].append(sn_table[i]['weight_1'])
                if t[n] > max_sol:
                    max_sol = t[n]
                    id = n

    #    for i in range(len(t)):
    #        print sources[sid[i]-1],'Sol: ',t[i], 'QF: ',max(1/(qf[i]-0.00001),0), tr[i]
    #    print 'Max sol:',max_sol

    if do_write == 1:
        file = './' + outname[0] + '/' + qualfile
        f = open(file, 'w')
        for i in range(len(t)):
            f.writelines(' ' + sources[sid[i] - 1] + ' Sol: ' + str(t[i]) + ' QF: ' + str(
                round(max(1 / (qf[i] - 0.00001), 0), 3)) + ' ' + str(tr[i]) + '\n')
        f.close()

    scan = 0
    good_scans = []
    bad_scans = []
    bad_sources = []
    for i in range(len(t)):
        if t[i] == max_sol:
            good_scans.append(i)
        elif t[i] < max_sol * 0.4:
            bad_scans.append(i)

    scan = good_scans[0]
    source = sources[sid[0] - 1]
    timerange = tr[0]

    for i in good_scans:
        if qf[i] <= qf[scan]:
            scan = i
            source = sources[sid[i] - 1]
            timerange = tr[i]

    for i in range(len(bad_scans)):
        k = bad_scans[i]
        bad_sources.append(sources[sid[k] - 1])
    logging.info('####################################')
    logging.info('Bad sources: ', str(bad_sources))
    logging.info('####################################')

    logging.info('####################################')
    logging.info('Manual phase-cal on: ',  source)
    logging.info('####################################')

    max_sol = naxis[3] * naxis[1] * len(indata.antennas)

    logging.info('####################################')
    logging.info('TIMERANGE: ', str(timerange) )
    logging.info('Max number of solutions: ',  str(max_sol))
    logging.info('####################################')

    return source, timerange
def get_sources(indata):
    su_table = indata.table('AIPS SU', 0)
    max_source = 0

    for i in su_table:
        if i['id__no'] > max_source:
            max_source = i['id__no']

    sources = []
    for i in range(max_source):
        sources.append([])

    for i in su_table:
        sources[i['id__no'] - 1] = i['source']
    return sources

def get_timerange_tab(indata,table,i):
    sn_table = indata.table(table, 0)
    time1 = sn_table[i].time-0.5*sn_table[i].time_interval
    time2 = sn_table[i].time+0.5*sn_table[i].time_interval
    (day1,hour1,min1,sec1)=time_to_hhmmss(time1)
    (day2,hour2,min2,sec2)=time_to_hhmmss(time2)
    timerange = [day1, hour1, min1, sec1, day2, hour2, min2, sec2]
    return timerange

def time_to_hhmmss(time):
    day = int(time)
    if time > 1:
        time = time - int(time)
    hour = int(time * 24)
    min = int(60 * (time * 24 - hour))
    sec = int(60 * (60 * (time * 24 - hour) - min))
    return day, hour, min, sec

def delete_temp():
    if os.path.exists('tmp_test*.txt'):
        os.remove('tmp_test*.txt')
    if os.path.exists('tmp_test*.txt'):
        os.remove('tmp_test*.txt')
