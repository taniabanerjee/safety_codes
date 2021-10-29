import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import logging
logging.basicConfig(level=logging.INFO)
import pymysql
from joblib import Parallel, delayed
import itertools
import datetime
from datetime import timedelta
import sys
from queuelib import FifoDiskQueue
import json
from scipy import stats
from ast import literal_eval
from shapely.geometry import Point, Polygon

configfile = sys.argv[1]

dboptions = {}
def extractOptions(fp):
   line = fp.readline().rstrip('\n')
   while (line != ''):
       strlist = line.split()
       dboptions[strlist[0]] = strlist[1]
       if (len(strlist) > 2):
           dboptions[strlist[0]] = strlist[1] + ' ' + strlist[2]
       line = fp.readline().rstrip('\n')
   line = fp.readline().rstrip('\n')
   return line

filepath = configfile
with open(filepath) as fp:
   line = fp.readline().rstrip('\n')
   while (line):
       line = extractOptions(fp)

def speeds_parallel(df_t):  # Why is this considered parallel?
    df_t = df_t.sort_values(by='timestamp')
    df_t = df_t.drop_duplicates(subset=['center_x', 'center_y'], keep = 'first').reset_index(drop = True)
    df_t['x'] = (round(df_t.center_x/2.0)*2).astype(int)
    df_t['y'] = (round(df_t.center_x/2.0)*2).astype(int)
    df_t = df_t.drop_duplicates(subset=['x', 'y'], keep = 'first').reset_index(drop = True)

#    x = round(df_t.center_x).astype(int)
#    y = round(df_t.center_y).astype(int)
#    coord_array = np.array(list(zip(x,y)))
#    values, count = stats.mode(coord_array)
#    while (count[0][0] > 2 and count[0][1] > 2):
#        index_names = df_t[(x == values[0][0]) & (y==values[0][1])].index
#        df_t = df_t.drop(index_names)
#        x = round(df_t.center_x).astype(int)
#        y = round(df_t.center_y).astype(int)
#        coord_array = np.array(list(zip(x,y)))
#        values, count = stats.mode(coord_array)

    df_t['a'] = df_t['center_x'].diff()
    df_t['b'] = df_t['center_y'].diff()
    df_t['td'] = df_t['timestamp'].diff().dt.total_seconds()

    df_t['a'] = df_t['a'].fillna(0)
    df_t['b'] = df_t['b'].fillna(0)
    df_t['td'] = df_t['td'].fillna(.1)

    df_t['speed_x'] = df_t['a']/df_t['td']
    df_t['speed_y'] = df_t['b']/df_t['td']

    #compute and convert speeds
    df_t['inst_speed'] =  np.sqrt((df_t['speed_x'])*(df_t['speed_x']) + (df_t['speed_y'])*(df_t['speed_y']))
    #Need to convert this to pixels per second
    #df_t['inst_speed'] = df_t['inst_speed'] #* 3 * 2.23694/50
    #df_t['inst_speed'] = df_t['inst_speed'] / (3 * 2.23694/50)
    df_t['absspeed'] = df_t['inst_speed'].rolling(window=5).mean()
    df_t['absspeed'] = df_t['absspeed'].fillna(0)
    df_t['speeddiff'] = df_t['absspeed'].diff()
    df_t['speeddiff'] = df_t['speeddiff'].fillna(0)
    
    #accleration
    df_t['accleration'] = df_t['speeddiff']/df_t['td']
 
    #skip the first two entries since these are needed to set up abs speed
    return df_t[2:] 

# def find_intersection(phase1, phase2, threshold=10000): # d < 14.45653732 is where most intersection tracks vary in centroid distance
#     intersection = ()
#     distances = []  # For testing purposes
#     for tup1 in phase1:
#         for tup2 in phase2:
#             dist = sum((c1-c2)**2 for c1,c2 in zip(tup1, tup2))**(1/2)
#             distances.append(dist)
#             if dist < threshold:
#                 threshold = dist
#                 intersection = tuple(((c1+c2)/2) for c1,c2 in zip(tup1, tup2))
#     return intersection if intersection !=() else None, distances

def getCosineOfAngle(Axi, Ayi, Bxi, Byi):
    a = 0
    if ((Axi != 0 or Ayi != 0) and (Bxi != 0 or Byi != 0)):
        [nAxi, nAyi] = 1/np.sqrt(Axi*Axi + Ayi*Ayi) * np.array([Axi, Ayi])
        [nBxi, nByi] = 1/np.sqrt(Bxi*Bxi + Byi*Byi) * np.array([Bxi, Byi])
        a = nAxi*nBxi + nAyi*nByi
    return a

def inLine(cx1, cy1, vx1, vy1, x1, y1):
    status = False
    cosvalue = getCosineOfAngle(vx1, vy1, (x1-cx1), (y1-cy1))
    if cosvalue > 0.95:
        status = True
    return status

#from https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    try:
       px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
       py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
       return [px, py]
    except:
       px = 0
       py = 0
       return [px, py]

def findTTC(cx1, cy1, cx2, cy2, vx1, vy1, vx2, vy2):
    rx1 = -100
    ry1 = -100
    rt1 = -100
    rt2 = -100
    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    v1 = np.sqrt(vx1**2 + vy1**2)
    v2 = np.sqrt(vx2**2 + vy2**2)
    #Check if velocity is 0 for objects 1 and/or 2
    if (v1 == 0 and v2 == 0):
        return rx1, ry1, rt1, rt2
    elif (v1 == 0):
        if (inLine(cx1, cy1, vx1, vy1, cx2, cy2)):
            return cx2, cy2, dist/v1, dist/v1
    elif (v2 == 0):
        if (inLine(cx2, cy2, vx2, vy2, cx1, cy1)):
            return cx1, cy1, dist/v2, dist/v2
    else:
        #Compute cosine of the angle between the two velocity vectors
        V = [(vx2-vx1), (vy2-vy1)]
        D = [(cx2-cx1), (cy2-cy1)]
        cosav = getCosineOfAngle(V[0], V[1], D[0], D[1])
        if (abs(cosav) > 0.95):
            ttc = dist/(v2-v1)  #check
            conflict_x = cx2 + vx2 * ttc
            conflict_y = cy2 + vy2 * ttc
            return conflict_x, conflict_y, ttc, ttc
        else:
            x1 = cx1
            y1 = cy1
            x2 = cx1 + vx1 #set t=1
            y2 = cy1 + vy1
            x3 = cx2
            y3 = cy2
            x4 = cx2 + vx2
            y4 = cy2 + vy2
            conflict_x, conflict_y = findIntersection(x1,y1,x2,y2,x3,y3,x4,y4)
            if (conflict_x == 0 and conflict_y == 0):
                return rx1, ry1, rt1, rt2
            if (inLine(cx1, cy1, vx1, vy1, conflict_x, conflict_y) and
                    inLine(cx2, cy2, vx2, vy2, conflict_x, conflict_y)):
                d1 = np.sqrt((conflict_x - cx1)**2 + (conflict_y - cy1)**2)
                ttc1 = d1/v1
                d2 = np.sqrt((conflict_x - cx2)**2 + (conflict_y - cy2)**2)
                ttc2 = d2/v2
                if (abs(ttc1-ttc2) < 10):
                    return conflict_x, conflict_y, ttc1, ttc2
            return rx1, ry1, rt1, rt2

    return rx1, ry1, rt1, rt2

def TTI(df_gppy, cpi_dict, gp, vehspat, vphasetime, mdrac, threshold=10): # , cpi_frame=None
    conflicting_phases = {
            1: [4,8,3,7,2],
            2: [4,8,3,7,1],
            3: [4,5,1,2,6],
            4: [6,3,5,2,1],
            5: [4,3,7,6,8],
            6: [8,5,7,4,3],
            7: [6,5,1,8,2],
            8: [2,7,1,6,5]
    }

    ped_conflicting_phases = {
            2: [4,8,3,1],
            4: [6,3,5,2],
            6: [8,5,7,4],
            8: [2,7,1,6]
    }

    list_ = []
    for i,rowi in df_gppy.iterrows():
        for j,rowj in df_gppy.iterrows():
            ## Continue Cases ##
            try:
                rowi.phase, rowj.phase = int(rowi.phase), int(rowj.phase)
            except ValueError:
                continue
            if i>=j:  # Already compared
                continue
            if rowi.phase < 0 or rowi.phase > 18 or rowj.phase < 0 or rowj.phase > 8:
                continue
            if rowj['class'] != 'pedestrian':
                if rowi.phase not in conflicting_phases[rowj.phase]:
                    continue
            else:
                if rowi.phase not in ped_conflicting_phases[rowj.phase]:
                    continue
            if rowi['intersection_id'] != rowj['intersection_id']:  # Not same intersection
                continue
            if not abs(rowj.timestamp - rowi.timestamp) < timedelta(seconds=10):  # Too far apart temporally
                continue
            if rowi['unique_ID'] == rowj['unique_ID']:  # Self comparison
                continue
            a, b = [1 if x['class'] == 'pedestrian' else 0 for x in (rowi, rowj)]
            p2v = a + b
            if p2v == 2:  # p2p
                continue
            
            ## Shorten Variable Names ##
            cx1, cy1, cx2, cy2 = rowi['center_x'], rowi['center_y'], rowj['center_x'], rowj['center_y']
            vx1, vy1, vx2, vy2 = rowi['speed_x'], rowi['speed_y'], rowj['speed_x'], rowj['speed_y']
#             ax1, ay1, ax2, ay2 = rowi['accleration'], rowi['accleration'], rowj['accleration'], rowj['accleration']
            
            ## Compute Metrics ##
            diffabs = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            distance_threshold_pixels = 10 * 50/3
            sp1, sp2 = rowi['inst_speed'] * (3 * 2.23694/50), rowj['inst_speed'] * (3 * 2.23694/50)
            speed_threshold = 2
            
#            if (sp1 < speed_threshold or sp2 < speed_threshold):
#                print ('Speed Threshold')
#                continue
#            if diffabs > distance_threshold_pixels:
#                print ('Distance Threshold')
#                continue
             
            dfobj1, dfobj2 = gp.get_group(rowi['unique_ID']), gp.get_group(rowj['unique_ID'])
            accel1value, accel2value, accel1time, accel2time = findDeceleration(dfobj1, dfobj2, rowi.frame_id, rowj.frame_id)
            conflict_x, conflict_y, t1, t2 = findTTC(cx1, cy1, cx2, cy2, vx1, vy1, vx2, vy2) 
            #t2 = (vx1 * cy1 - vx1 * cy2 + vy1 * cx2 - vy1 * cx1) / ((vx1 * vy2 - vx2* vy1) + 0.001)  # X plugged into Y
            #t1 = (cx2 - cx1 + vx2 * t2) / (vx1 + 0.001) if vx1 != 0 else (cy2 - cy1 + vy2 * t2) / (vy1 + 0.001)
            #conflict_x, conflict_y = (cx1 + vx1 * t1), (cy1 + vy1 * t1)
           
            ## Comparisons Not of Interest ##
            if (conflict_x < 0 or conflict_x > 1279 or conflict_y < 0 or conflict_y > 959):  # Conflict Outside Intersection
                continue
            if ((t1+t2)/2 > threshold):  # TTCs too Large
                continue
            if (t1<0) or (t2<0):  # Negative TTC's
                continue
            ## Compute Metrics pt.2 ##
            sp1mps = sp1 * 0.44704 # mph -> mps
            sp2mps = sp2 * 0.44704 # mph -> mps
            diffabs1 = np.sqrt((cx1-conflict_x)**2 + (cy1-conflict_y)**2)
            diffabs2 = np.sqrt((cx2-conflict_x)**2 + (cy2-conflict_y)**2)
            drac1 = (sp1mps**2)/(2*diffabs1)  # V^2/2s
            drac2 = (sp2mps**2)/(2*diffabs2)
            madr1 = mdrac[rowi['class']]*3/50 # deceleration CPI FORMULA
            madr2 = mdrac[rowj['class']]*3/50
            cpi1, cpi2 = [int(d>m) for d, m in [(drac1, madr1), (drac2, madr2)]]
            cp1, cp2 = cpi1*rowi['td'], cpi2*rowj['td']
            for ID, t, c in [(rowi['unique_ID'],rowi['timestamp'],cp1), (rowj['unique_ID'],rowj['timestamp'],cp2)]:
                total = cpi_dict[ID]['total']
                c = c / total if total != 0 else 0
                cpi_dict[ID]['cpi'] = cpi_dict[ID]['cpi'] + c
            
            ## Encode Table ##
            d = {}
            d['intersection_id'] = rowi['intersection_id']
            d['camera_id'] = rowi['camera_id']
            d['timestamp'] = rowi['timestamp']
            d['dow'] = rowi.timestamp.weekday()
            d['hod'] = rowi.timestamp.hour
            d['frame_id'] = rowi['frame_id']
            d['conflict_x'], d['conflict_y'] = conflict_x, conflict_y
            d['unique_ID1'], d['unique_ID2'] = rowi['unique_ID'], rowj['unique_ID']
            d['class1'], d['class2'] = rowi['class'], rowj['class']
            d['phase1'],   d['phase2'] = rowi.phase, rowj.phase
            d['time'] = abs(t2-t1)
            d['p2v'] = p2v
            d['city'] = rowi['city']
            d['cluster1'], d['cluster2'] = rowi['cluster'], rowj['cluster']
            d['speed1'] = sp1
            d['speed2'] = sp2
            d['distance'] = diffabs
            d['deceleration1'] = abs(accel1value)
            d['deceleration2'] = abs(accel2value)
            d['decel1_ts'] = accel1time
            d['decel2_ts'] = accel2time
            d['type'] = 1 #TTC
            d['signal_state'] = rowi['SPAT']
            vinterval, vexactmatch = findPhaseDuration(vehspat, vphasetime, rowi.timestamp)
            d['phase_duration'] = vinterval
            relative_time = (rowi.timestamp - vexactmatch.timestamp).total_seconds()
            d['percent_in_phase'] = relative_time*100/vinterval
            list_.append(d)           
    return list_

def findPhaseDuration(vehspat, vphasetime, ts):
    vindex = np.searchsorted(vehspat.timestamp, ts)
    if (vindex == 0):
        vexactmatch = vehspat.iloc[0]
    else:
        vexactmatch = vehspat.iloc[vindex-1]
    try:
        vinterval = vphasetime.iloc[vindex]
    except:
        vinterval = 0
    return vinterval, vexactmatch

def getIntervalIndex(xlist, xgrid):
    output = []
    for x in xlist:
        s = xgrid-x
        igt = np.where(s>0)[0]
        xindex = 0
        if (len(igt) > 0):
            xindex = igt[0]
        output.append(xindex)

    return output

def compute_gaps(tracks, mydb, flat_list, vehspat, vphasetime, mdrac, cpi_dict):
    gapdict = {}
    #create a mesh grid
    #for each trajectory get the grid cells that are getting activated
    #for each track, get the  compute gaps with the next vehicle

    xmin = min([x for (x,y) in flat_list])
    ymin = min([y for (x,y) in flat_list])
    xmax = max([x for (x,y) in flat_list])
    ymax = max([y for (x,y) in flat_list])

    xgrid = np.linspace(xmin, xmax, 50)
    ygrid = np.linspace(ymin, ymax, 50)
    xgrid = np.asarray([round(num,1) for num in xgrid])
    ygrid = np.asarray([round(num,1) for num in ygrid])

    tlist = tracks['unique_ID'].unique()
    for unique_id in tlist:
        rows = tracks.loc[tracks['unique_ID'] == unique_id]
        x = rows['center_x'].tolist()
        y = rows['center_y'].tolist()
        t = rows['timestamp'].tolist()
        x_index = getIntervalIndex(x, xgrid)
        y_index = getIntervalIndex(y, ygrid)
        carray = np.array([[xx, yy] for xx, yy in zip(x_index, y_index)])
        for i in range(len(carray)):
            c = tuple(carray[i])
            ts = t[i]
            if (gapdict.get(c) == None):
                gapdict[c] = []
                gapdict[c].append([unique_id, ts, ts])
            else:
                value = gapdict[c]
                found = 0
                for v in value:
                    if v[0] == unique_id:
                        v[2] = ts
                        found = 1
                        break
                if (found == 0):
                    gapdict[c].append([unique_id, ts, ts])
    gaplist = {}
    dflist = []

    #values = gapdict.values()
    for key,vu in gapdict.items():
        v = sorted(vu, key=lambda x: x[1])
        for i in range(0, len(v)):
            if (i+1 < len(v)):
                gap = (v[i+1][1] - v[i][2]).total_seconds()
                if (gap > 20 or gap < 0):
                    continue
                vistr = str(v[i][0])
                vnistr = str(v[i+1][0])
                if (vistr[0:2] != vnistr[0:2]):
                    continue
                if gaplist.get(v[i+1][0]) == None:
                    gaplist[v[i+1][0]] = []
                gaplist[v[i+1][0]].append(gap)
                if (gap < 10):
                    track1 = tracks[tracks.unique_ID==v[i][0]]
                    track2 = tracks[tracks.unique_ID==v[i+1][0]]
                    cluster1 = track1.cluster.iloc[0]
                    cluster2 = track2.cluster.iloc[0]
                    if (cluster1 != cluster2):
                        #record PET
                        d = {}
                        d['intersection_id'] = track1.intersection_id.iloc[0]
                        d['camera_id'] = track1.camera_id.iloc[0]
                        d['timestamp'] = v[i][2]
                        d['dow'] = v[i][2].weekday()
                        d['hod'] = v[i][2].hour
                        d['conflict_x'], d['conflict_y'] = xgrid[key[0]], ygrid[key[1]]
                        d['unique_ID1'], d['unique_ID2'] = track1.unique_ID.iloc[0], track2.unique_ID.iloc[0]
                        d['class1'], d['class2'] = track1['class'].iloc[0], track2['class'].iloc[0]
                        d['phase1'],   d['phase2'] = track1.phase.iloc[0], track2.phase.iloc[0]
                        d['type'] = 0 #PET
                        d['time'] = gap
                        d['p2v'] = 0
                        if (d['class1'] == 'pedestrian' and d['class2'] != 'pedestrian' or d['class1'] != 'pedestrian' and d['class2'] == 'pedestrian'):
                            d['p2v'] = 1
                        d['city'] = track1.city.iloc[0]
                        d['cluster1'], d['cluster2'] = cluster1, cluster2
                        try:
                            d['speed1'] = track1[track1['timestamp'] == v[i][2]].inst_speed.item()
                            d['signal_state'] =  track1[track1['timestamp'] == v[i][2]].SPAT.item()
                            x1 = track1[track1['timestamp'] == v[i][2]].center_x.item()
                            y1 = track1[track1['timestamp'] == v[i][2]].center_y.item()
                            frame_id1 = track1[track1['timestamp'] == v[i][2]].frame_id.item()
                        except:
                            d['speed1'] = np.mean(track1.absspeed)
                            d['signal_state'] = track1.iloc[0].SPAT
                            x1 = track1.iloc[0].center_x
                            y1 = track1.iloc[0].center_y
                            frame_id1 = track1.iloc[0].frame_id.item()
                        try:
                            d['speed2'] = track2[track2['timestamp'] == v[i+1][1]].inst_speed.item()
                            frame_id2 = track2[track2['timestamp'] == v[i+1][1]].frame_id.item()
                        except:
                            d['speed2'] = np.mean(track2.absspeed)
                            frame_id2 = track2.iloc[0].frame_id.item()
                        try:
                            vindex = np.searchsorted(track2.timestamp, v[i][2])
                            x2 = track2[vindex-1].center_x.item()
                            y2 = track2[vindex-1].center_y.item()
                        except:
                            x2 = track2.iloc[0].center_x
                            y2 = track2.iloc[0].center_y
                        d['distance'] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        accel1value, accel2value, accel1time, accel2time = findDeceleration(track1, track2, frame_id1, frame_id2)
                        d['deceleration1'] = abs(accel1value)
                        d['deceleration2'] = abs(accel2value)
                        d['decel1_ts'] = accel1time
                        d['decel2_ts'] = accel2time
                        vinterval, vexactmatch = findPhaseDuration(vehspat, vphasetime, v[i][2])
                        d['phase_duration'] = vinterval
                        relative_time = (v[i+1][1] - vexactmatch.timestamp).total_seconds()
                        d['percent_in_phase'] = relative_time*100/vinterval
                        drac1 = d['deceleration1']
                        drac2 = d['deceleration2']
                        madr1 = mdrac[track1.iloc[0]['class']]*3/50 # deceleration CPI FORMULA
                        madr2 = mdrac[track2.iloc[0]['class']]*3/50
                        cpi1, cpi2 = [int(d>m) for d, m in [(drac1, madr1), (drac2, madr2)]]
                        try:
                            time1 = track1[track1.accleration == drac1].iloc[0].td
                        except:
                            time1 = 0
                        try:
                            time2 = track2[track2.accleration == drac2].iloc[0].td
                        except:
                            time2 = 0
                        cp1, cp2 = cpi1*time1, cpi2*time2
                        for ID, c in [(track1.iloc[0].unique_ID,cp1), (track2.iloc[0].unique_ID,cp2)]:
                            total = cpi_dict[ID]['total']
                            c = c / total if total != 0 else 0
                            cpi_dict[ID]['cpi'] = cpi_dict[ID]['cpi'] + c
            
                        dflist.append(d)
    gapdict.clear()
    if dflist:
        cgaps = pd.DataFrame(dflist)
        cursor=mydb.cursor()
        cols = "`,`".join([str(i) for i in cgaps.columns.tolist()])
        sql = "INSERT INTO `PETtable` (`" +cols + "`) VALUES (" + "%s,"*(cgaps.shape[1]-1) + "%s)"
        logging.info("Inserting records to database")
        cgaps.timestamp1 = cgaps.timestamp1.astype(str)
        cgaps.timestamp2 = cgaps.timestamp2.astype(str)
        tuples = [tuple(x) for x in cgaps.to_numpy()]
        cursor.executemany(sql, tuples)
        mydb.commit()
        cursor.close()

    return dflist

def findDeceleration(dfobj1, dfobj2, frame_id1, frame_id2):
    start_frame = frame_id1 - 5
    last_frame = frame_id2 + 2
    acc1 = dfobj1[(dfobj1.frame_id>start_frame)&(dfobj1.frame_id<=last_frame)]
    start_frame = frame_id2 - 5
    last_frame = frame_id2 + 2
    acc2 = dfobj2[(dfobj2.frame_id>start_frame)&(dfobj2.frame_id<=last_frame)]
    accel1 = acc1[acc1.accleration < 0]
    accel2 = acc2[acc2.accleration < 0]
#     if (accel1.empty and accel2.empty):
#         print ('Acceleration Threshold')
#         continue

    accel1value = 0
    accel2value = 0
    accel1time = str(0)
    accel2time = str(0)
    if (accel1.empty == False):
        accel1value = accel1.iloc[accel1['accleration'].argmin()].accleration
        t = accel1.iloc[accel1['accleration'].argmin()].timestamp
        accel1time = t.strftime("%Y-%m-%d %H:%M:%S.%f")
    if (accel2.empty == False):
        accel2value = accel2.iloc[accel2['accleration'].argmin()].accleration
        t = accel2.iloc[accel2['accleration'].argmin()].timestamp
        accel2time = t.strftime("%Y-%m-%d %H:%M:%S.%f")
    return accel1value, accel2value, accel1time, accel2time

def getMedian(df):
    phases = [0] * df.shape[0]
    phase1 = df.phase1
    phase2 = df.phase2
    phases = [2 if phase1.iloc[i] == 5 or phase1.iloc[i] == 7 or phase2.iloc[i] == 5 or phase2.iloc[i] == 7 else phases[i] for i in range(df.shape[0])]
    phases = [6 if phase1.iloc[i] == 1 or phase1.iloc[i] == 3 or phase2.iloc[i] == 1 or phase2.iloc[i] == 3 else phases[i] for i in range(df.shape[0])]
    median = [median_width[i] for i in phases]
    return median

def getTotalVehicles(track_p, df):
    tslist = df['timestamp']
    start_time = tslist-datetime.timedelta(seconds = 30)
    end_time = tslist+datetime.timedelta(seconds = 30)
    number_vehicles = [track_p[(track_p['timestamp'] > s) & (track_p['timestamp'] < e)].shape[0] for s, e in zip(start_time, end_time)]
    return number_vehicles

def do_main(body):
    (intersection_id,camera_id,start_time,end_time)=json.loads(body)
#     threshold = 11
    
    ################################################ GET TRACKS #########################################################
    #print("Querying Tracks")
    mydb = pymysql.connect(host='maltlab.cise.ufl.edu', user='root', password='maltserver', database='testdb')
    spat_start = pd.to_datetime(start_time) - datetime.timedelta(minutes=30)
    spat_end = pd.to_datetime(end_time) + datetime.timedelta(minutes=30)
    spat_query  = "SELECT * FROM OnlineSPaT where intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,spat_start.strftime("%Y-%m-%d %H:%M:%S"),spat_end.strftime("%Y-%m-%d %H:%M:%S"))

    spatdf = pd.read_sql(spat_query, con=mydb)
    vehspat = spatdf[spatdf['type']==1]
    vphasetime = vehspat.timestamp.diff().dt.total_seconds()

    intersectionquery = 'SELECT * from IntersectionProperties where intersection_id={} and camera_id={};'.format(intersection_id, camera_id)
    intersectionprop = pd.read_sql(intersectionquery, con=mydb)
    median_width = {}
    median_width[2] = intersectionprop['median_2'][0]
    median_width[4] = intersectionprop['median_4'][0]
    median_width[6] = intersectionprop['median_6'][0]
    median_width[8] = intersectionprop['median_8'][0]
    median_width[0] = 0
    poly = intersectionprop['polygon'][0]
    polylist = [literal_eval(poly)]
    flat_list = [item for sublist in polylist for item in sublist]
    polygon = Polygon(flat_list)

    box = polygon.minimum_rotated_rectangle
    x, y = box.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    length = max(edge_length)
    width = min(edge_length)
    mbr = np.sqrt(length*length + width*width)

    sql_query  = "SELECT * FROM RealDisplayInfo where intersection_id = {}  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,start_time,end_time)
    tracks = pd.read_sql(sql_query, con=mydb)
    tracks['timestamp'] = pd.to_datetime(tracks['timestamp'])
    #print(f"Done loading /'tracks/'. No of tracks are {tracks.shape[0]}")
    
    sql_query  = "SELECT * FROM RealTrackProperties where intersection_id = {}  AND timestamp >= '{}' AND timestamp <= '{}' and (isAnomalous != 1 or lanechange=1)".format(intersection_id,start_time,end_time)
    track_p = pd.read_sql(sql_query, con=mydb)
    track_p['timestamp'] = pd.to_datetime(track_p['timestamp'])
    #print(f"Done loading /'track_p/'. No of tracks are {track_p.shape[0]}")
    
    sql_query  = "SELECT * FROM TracksReal where intersection_id = {}  AND start_timestamp >= '{}' AND start_timestamp <= '{}'".format(intersection_id,start_time,end_time)
    track_s = pd.read_sql(sql_query, con=mydb)
    track_s['start_timestamp'] = pd.to_datetime(track_s['start_timestamp'])
    #print(f"Done loading /'track_s/'. No of tracks are {track_s.shape[0]}")
    ######################################################################################################################
    #Skip Odd Tracks
    tracks1 = tracks[tracks['skip_begin'] != 1].copy()
    tracks1 = tracks1[tracks1['skip_end'] != 1]
    #tracks1 = tracks1[tracks1['skip_mid'] != 1]
    #tracks1 = tracks1[tracks1['skip_angle'] != 1]
    #tracks['timestamp'].dt.date.unique()
    #Skip out of bounds points
    tracks1  = tracks1[(tracks1.center_x > 0 ) & (tracks1.center_x<1280)]
    tracks1  = tracks1[(tracks1.center_y > 0 ) & (tracks1.center_y<960)]

    #apply function to compute speeds and accleration
    ############################################### COMPUTE SPEEDS PARALLEL ###############################################
    #print("Computing speeds")
    gpby = tracks1.groupby(['unique_ID'])
    #processed_list = Parallel(n_jobs=num_cores)(delayed(speeds_parallel)(i) for s,i in gpby)
    processed_list = []
    for s,i in gpby:
        processed_list.append(speeds_parallel(i))
    try:
        tracks1 = pd.concat(processed_list)
    except:
        return

    temp = tracks1.join(track_p[['unique_ID', 'phase', 'cluster', 'redJump','lanechange','nearmiss', 'city']].set_index('unique_ID'), on='unique_ID')
    cgpby = temp.groupby(['frame_id'])

    cpi_dict = {}
    gp = temp.groupby(['unique_ID'])
    for uid,rows in gp:
        firstindex = 0
        lastindex = np.shape(rows)[0]-1
        tslist = rows['timestamp'].to_list()
        start_time = tslist[firstindex]
        end_time = tslist[lastindex]
        total_time = (end_time - start_time).total_seconds()
        cpi_dict[uid] = {'cpi': 0, 'total': total_time}
    #print("Computing pets")
    mdrac = {'car': 50, 'truck': 50, 'semi': 2,'bus': 5, 'pedestrian': 100, 'motorbike': 75, 'vehicle' : 50, 'vehicle': 50}
    petlist = compute_gaps(temp, mydb, flat_list, vehspat, vphasetime, mdrac, cpi_dict)
    ############################################## COMPUTE TTCs ###########################################################
    #processed_list = Parallel(n_jobs=num_cores)(delayed(TTI)(l, intersections, threshold) for _,l in cgpby)
    # Process tracks
    num_cores = 10
    processed_list = Parallel(n_jobs=num_cores)(delayed(TTI)(l, cpi_dict, gp, vehspat, vphasetime, mdrac) for _,l in cgpby)
    #for indexname, rows in cgpby:
        #TTI(rows, cpi_dict, gp, vehspat, vphasetime, mdrac)
    cflat_list = [item for sublist in processed_list for item in sublist]
    if (len(cflat_list) > 0):
        cgaps = pd.DataFrame(cflat_list)
        cpi1 = [cpi_dict[uid]['cpi'] for uid in cgaps.unique_ID1]
        cpi2 = [cpi_dict[uid]['cpi'] for uid in cgaps.unique_ID2]
        cgaps['cpi1'] = cpi1
        cgaps['cpi2'] = cpi2
        conflictsgpby = cgaps.groupby(['unique_ID1', 'unique_ID2'])
        unique_conflicts = pd.DataFrame(columns=cgaps.columns)
        for index, rows in conflictsgpby:
            unique_conflicts = unique_conflicts.append(rows[rows.ttc == rows.ttc.min()])
        conflictsgpby = unique_conflicts.groupby(['unique_ID1'])
        uid1_conflicts = pd.DataFrame(columns=cgaps.columns)
        count1 = {}
        for index, rows in conflictsgpby:
            uid1_conflicts = uid1_conflicts.append(rows[rows.ttc == rows.ttc.min()])
            count1[index] = rows.shape[0]
        uid2_conflicts = pd.DataFrame(columns=cgaps.columns)
        conflictsgpby = unique_conflicts.groupby(['unique_ID2'])
        count2 = []
        for index, rows in conflictsgpby:
            uid2_conflicts = uid2_conflicts.append(rows[rows.ttc == rows.ttc.min()])
            c2 = rows.shape[0]
            if (count1.get(index)):
                c2 = c2 + count1[index]
            count2.append(c2)
        uid2_conflicts['num_involved'] = count2
        uid2_conflicts['intersection_diagonal'] = [mbr] * uid2_conflicts.shape[0]
        uid2_conflicts['median_width'] = getMedian(uid2_conflicts)
        uid2_conflicts['total_vehicles'] = getTotalVehicles(track_p, uid2_conflicts)

        mydb = pymysql.connect(host='maltlab.cise.ufl.edu', user='root', password='maltserver', database='testdb')
        cursor=mydb.cursor()
        cols = "`,`".join([str(i) for i in uid2_conflicts.columns.tolist()])
        sql = "INSERT INTO `TTCTable` (`" +cols + "`) VALUES (" + "%s,"*(uid2_conflicts.shape[1]-1) + "%s)"
        logging.info("Inserting records to database")
        uid2_conflicts.timestamp = uid2_conflicts.timestamp.astype(str)
        tuples = [tuple(x) for x in uid2_conflicts.to_numpy()]
        cursor.executemany(sql, tuples)
        mydb.commit()
        cursor.close()

    if (len(petlist) > 0):
        cflat_list = [item for sublist in petlist for item in sublist]
        cgaps = pd.DataFrame(cflat_list)
        cpi1 = [cpi_dict[uid]['cpi'] for uid in cgaps.unique_ID1]
        cpi2 = [cpi_dict[uid]['cpi'] for uid in cgaps.unique_ID2]
        cgaps['cpi1'] = cpi1
        cgaps['cpi2'] = cpi2
        cgaps['num_involved'] = [0] * cgaps.shape[0]
        cgaps['intersection_diagonal'] = [mbr] * cgaps.shape[0]
        cgaps['median_width'] = getMedian(cgaps)
        cgaps['total_vehicles'] = getTotalVehicles(track_p, cgaps)


if __name__ == "__main__":
    argdf = pd.read_csv('2175.csv', header=None)
    arglist = argdf.to_numpy().tolist()
    arglist = [[2175, 15, ' 2021-10-13 17:10:00', ' 2021-10-13 17:20:00']]

    for cmd in arglist:
        print (cmd)
        do_main(json.dumps(cmd))

