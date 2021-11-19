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
import json
import time
from scipy import stats
from ast import literal_eval
from shapely.geometry import Point, Polygon
from TrajectorySmooth import TrajectorySmooth
import pika

if (len(sys.argv) == 1):
        print ('Error: provide config file')

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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def getPixelPerMeter():
    return 53/3.6576

def getMetersPerSecondToMilesPerHour():
    return 2.23694

def getPixelsPerSecondToMilesPerHour():
    return getMetersPerSecondToMilesPerHour()/getPixelPerMeter()

def speeds_parallel_old(df_t):  # Why is this considered parallel?
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
    df_t['absspeed'] = df_t['inst_speed'].rolling(window=3).mean()
    df_t['absspeed'] = df_t['absspeed'].fillna(0)
    df_t['speeddiff'] = df_t['absspeed'].diff()
    df_t['speeddiff'] = df_t['speeddiff'].fillna(0)
    
    #accleration
    df_t['accleration'] = df_t['speeddiff']/df_t['td']
 
    #skip the first two entries since these are needed to set up abs speed
    return df_t[2:] 

from numpy.linalg import norm
def find_angle(a, b, c):
    a = a-b; c = c-b
    mag = norm(a)*norm(b); num = np.dot(a, b)
    angle = np.arccos(num/mag)
    return angle*180.0/np.pi
def angle_diff_array(triplets):
    prev, angle_diffs = 0, []
    for (a,b,c) in triplets:
        curr = find_angle(a,b,c)
        angle_diffs.append(prev-curr)
        prev = curr
    return np.array(angle_diffs)
def track2angle_diffs(track):
    period, stop = 3, len(track)
    triplets = np.zeros((stop,3,2))
    for i in range(stop):
        if i == 0:  # Beginning
            triplets[i] = np.array([[np.nan, np.nan], track[i], track[i+1]])
        elif i == stop-1:  # End
            triplets[i] = np.array([track[i-1], track[i], [np.nan, np.nan]])
        else:
            triplets[i] = np.array([track[i-1], track[i], track[i+1]])
    return angle_diff_array(triplets)
def rem_jitter_pipeline(angles, deviation=.5, limit=5):  # Could base by phase and its average turning
    mean = angles[~np.isnan(angles)].mean()
    std = angles[~np.isnan(angles)].std()
    degrees = std*deviation
    degrees = degrees if degrees>limit else limit
    nojit = np.array([])
    upper_threshold = mean+degrees
    lower_threshold = mean-degrees
    for i in range(len(angles)):
        if lower_threshold<=angles[i]<=upper_threshold:
            nojit = np.append(nojit, angles[i])
        else:
            nojit = np.append(nojit, np.nan)
    return nojit
def assign_angle_columns(temp):
    """Assumes temp is sorted by timestamp ascending"""
    u_groups = temp.groupby('unique_ID')
    for uid, frame in u_groups:
        if len(frame) < 4:
            continue
        angles = track2angle_diffs(frame[['center_x', 'center_y']].values)
        jitter_mask = rem_jitter_pipeline(angles)
        temp.loc[temp['unique_ID']==uid, 'valid_angle'] = jitter_mask
    temp = temp[temp['valid_angle'].notnull()]  # Remove points with invalid angles
    return temp

def speeds_parallel(df_t):  # Why is this considered parallel?
    df_t = df_t.sort_values(by='timestamp')
    df_t = df_t.drop_duplicates(subset=['center_x', 'center_y'], keep = 'first').reset_index(drop = True)
    df_t['valid_angle'] = np.nan
    if (df_t.shape[0] > 15):
        df_t = assign_angle_columns(df_t)

    df_t['a'] = df_t['center_x'].diff()
    df_t['b'] = df_t['center_y'].diff()
    df_t['td'] = df_t['timestamp'].diff().dt.total_seconds()

    df_t['a'] = df_t['a'].fillna(df_t.iloc[abs(df_t['a']).argmin()].a)
    df_t['b'] = df_t['b'].fillna(df_t.iloc[abs(df_t['b']).argmin()].b)
    df_t['td'] = df_t['td'].fillna(.1)

    df_t['speed_x'] = df_t['a']/df_t['td']
    df_t['speed_y'] = df_t['b']/df_t['td']

    #compute and convert speeds
    df_t['inst_speed'] =  np.sqrt((df_t['speed_x'])*(df_t['speed_x']) + (df_t['speed_y'])*(df_t['speed_y']))
    #Need to convert this to pixels per second
    #df_t['inst_speed'] = df_t['inst_speed'] #* 3 * 2.23694/50
    #df_t['inst_speed'] = df_t['inst_speed'] / (3 * 2.23694/50)
    df_t['absspeed'] = df_t['inst_speed'].rolling(window=3).mean()
    df_t['absspeed'] = df_t['absspeed'].fillna(df_t.iloc[abs(df_t['inst_speed']).argmin()].inst_speed)
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
    if abs(cosvalue) > 0.95:
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
    if (v1 < 6 and v2 < 6): #speed in pixels per second
        return rx1, ry1, rt1, rt2
    elif (v1 < 6):
        if (inLine(cx1, cy1, vx1, vy1, cx2, cy2)):
            return cx2, cy2, dist/v1, dist/v1
    elif (v2 < 6):
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
            return conflict_x, conflict_y, ttc, ttc+ttc #adding twice so that
                    #t2 - t1 in caller doesn't cause 0 ttc.
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

def phaseCheck(phase1, phase2, class1, class2, cluster1, cluster2, \
        conflicting_phases, ped_conflicting_phases):
    status = False  #don't ignore this combination of phase1 and phase2
    if (phase1 < 1 or phase1 > 8 or phase2 < 1 or phase2 > 8):
        status = True
    elif class1 != 'pedestrian' and class2 != 'pedestrian':
        if phase1 not in conflicting_phases[phase2]:
            status = True
            invalidCombinations = [['EBR', 'NBL'], ['WBR', 'SBL'], \
                    ['NBR', 'WBL'], ['SBR', 'EBL'], ['SBR', 'EBR'], \
                    ['EBR', 'NBR'], ['NBR', 'WBR'], ['WBR', 'SBR']]
            clus1 = cluster1[0:3]
            clus2 = cluster2[0:3]
            for c in invalidCombinations:
                if (clus1 in c and clus2 in c):
                    status = True
    elif class1 != 'pedestrian':
        if phase1 not in ped_conflicting_phases[phase2]:
            status = True
    elif class2 != 'pedestrian':
        if phase2 not in ped_conflicting_phases[phase1]:
            status = True
    else: # both are peds
            status = True
    return status

def TTI(df_gppy, gp, vehspat, vphasetime, conflicting_phases, \
        ped_conflicting_phases, threshold=10):
    list_ = []
    for i,rowi in df_gppy.iterrows():
        for j,rowj in df_gppy.iterrows():
            ## Continue Cases ##
            try:
                rowi.phase, rowj.phase = int(rowi.phase), int(rowj.phase)
            except ValueError:
                continue
            if rowi.track_id >= rowj.track_id:  # Already compared
                continue
            if rowi['intersection_id'] != rowj['intersection_id']:  # Not same intersection
                continue
            if (phaseCheck(rowi['phase'], rowj['phase'], rowi['class'], \
                    rowj['class'], rowi['cluster'], rowj['cluster'], \
                    conflicting_phases, ped_conflicting_phases)):
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
            distance_threshold_pixels = 10 * getPixelPerMeter()
            sp1 = rowi['inst_speed'] * getPixelsPerSecondToMilesPerHour()
            sp2 = rowj['inst_speed'] * getPixelsPerSecondToMilesPerHour()
            speed_threshold = 2
            
#            if (sp1 < speed_threshold or sp2 < speed_threshold):
#                print ('Speed Threshold')
#                continue
#            if diffabs > distance_threshold_pixels:
#                print ('Distance Threshold')
#                continue
             
            dfobj1, dfobj2 = gp.get_group(rowi['unique_ID']), gp.get_group(rowj['unique_ID'])
            accel1value, accel2value, accel1time, accel2time, accel1timeStr, accel2timeStr = findDeceleration(dfobj1, dfobj2, rowi.frame_id, rowj.frame_id)
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
            d['state'] = 'FL'
            d['cluster1'], d['cluster2'] = rowi['cluster'], rowj['cluster']
            d['speed1'] = sp1
            d['speed2'] = sp2
            d['distance'] = diffabs
            d['deceleration1'] = abs(accel1value)
            d['deceleration2'] = abs(accel2value)
            d['decel1_ts'] = accel1timeStr
            d['decel2_ts'] = accel2timeStr
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

def getTimeDiffIndex(trackdf, ts):
    idx = trackdf.index[trackdf.timestamp == ts].tolist()
    use_idx = idx[-1]
    idx_list = trackdf.index.tolist()
    index = idx_list.index(use_idx)
    return index

def compute_gaps(tracks, mydb, flat_list, vehspat, vphasetime, \
        conflicting_phases, ped_conflicting_phases):
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
        pixel_per_meter = getPixelPerMeter()
        xm, ym = np.array(rows.center_x)/pixel_per_meter, np.array(rows.center_y)/pixel_per_meter
        timestamps = np.array(rows.timestamp)
        #fit a spline and get missing grid cells.
        smooth_obj = TrajectorySmooth(xm, ym, timestamps=timestamps, smooth=0.85)
        abnormal_ratio, smoothed_xs, smoothed_ys = smooth_obj.smooth(back_thres=-0.1)
        nx_index, ny_index = smoothed_xs * pixel_per_meter, smoothed_ys * pixel_per_meter

        x_index = getIntervalIndex(nx_index.tolist(), xgrid)
        y_index = getIntervalIndex(ny_index.tolist(), ygrid)
        ts_index = pd.to_datetime(smooth_obj.interpolated_timestamps)
        carray = np.array([[xx, yy] for xx, yy in zip(x_index, y_index)])
        #smooth_obj = TrajectorySmooth(np.array(x_index), np.array(y_index), timestamps=np.array(np.array(rows.timestamp)), smooth=0.85)
        #abnormal_ratio, smoothed_xs, smoothed_ys = smooth_obj.smooth(back_thres=-0.1)
        #carray = np.array([[round(xx), round(yy)] for xx, yy in zip(smoothed_xs, smoothed_ys)])
        print (carray)
        for i in range(len(carray)):
            c = tuple(carray[i])
            ts = ts_index[i]
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
                #gap = (v[i+1][1] - v[i][2])/np.timedelta64(1, 's')
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
                    phase1 = track1.phase.iloc[0]
                    phase2 = track2.phase.iloc[0]
                    class1 = track1['class'].iloc[0]
                    class2 = track2['class'].iloc[0]
                    cluster1 = track1.cluster.iloc[0]
                    cluster2 = track2.cluster.iloc[0]
                    #get the record from tracks data that's closest to the
                    #timestamp from v
                    try:
                        r1 = track1.iloc[abs((track1['timestamp']-v[i][2])/np.timedelta64(1, 's')).argsort().iloc[0]]
                        r2 = track2.iloc[abs((track2['timestamp']-v[i+1][1])/np.timedelta64(1, 's')).argsort().iloc[0]]
                    except:
                        r1 = track1.iloc[track1.size-1]
                        r2 = track2.iloc[0]
                    if (phaseCheck(phase1, phase2, class1, class2, 
                            cluster1, cluster2, conflicting_phases, \
                                    ped_conflicting_phases)):
                        continue
                    vinterval, vexactmatch = findPhaseDuration(vehspat, \
                            vphasetime, v[i][2])
                    spat1 = vexactmatch.hexphase
                    vinterval, vexactmatch = findPhaseDuration(vehspat, \
                            vphasetime, v[i+1][1])
                    spat2 = vexactmatch.hexphase
                    if (spat1 != spat2):
                        continue
                    #record PET
                    d = {}
                    d['intersection_id'] = track1.intersection_id.iloc[0]
                    d['camera_id'] = track1.camera_id.iloc[0]
                    d['timestamp'] = v[i][2]
                    d['dow'] = v[i][2].weekday()
                    d['hod'] = v[i][2].hour
                    d['frame_id'] = r1.frame_id
                    d['conflict_x'] = xgrid[key[0]]
                    d['conflict_y'] = ygrid[key[1]]
                    d['unique_ID1'] = track1.unique_ID.iloc[0]
                    d['unique_ID2'] = track2.unique_ID.iloc[0]
                    d['class1'], d['class2'] = class1, class2
                    d['phase1'], d['phase2'] = phase1, phase2
                    d['time'] = gap
                    d['p2v'] = 0
                    if (class1 == 'pedestrian' and class2 != 'pedestrian' or \
                            class1 != 'pedestrian' and class2 == 'pedestrian'):
                        d['p2v'] = 1
                    d['city'] = track1.city.iloc[0]
                    d['state'] = 'FL'
                    d['cluster1'], d['cluster2'] = cluster1, cluster2
                    d['speed1'] = r1.inst_speed * \
                            getPixelsPerSecondToMilesPerHour()
                    d['speed2'] = r2.inst_speed * \
                            getPixelsPerSecondToMilesPerHour()
                    x1 = r1.center_x
                    y1 = r1.center_y
                    frame_id1 = r1.frame_id
                    x2 = r2.center_x
                    y2 = r2.center_y
                    frame_id2 = r2.frame_id
                    d['distance'] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    accel1value, accel2value, accel1time, accel2time, \
                            accel1timeStr, accel2timeStr = \
                            findDeceleration(track1, track2, frame_id1, \
                            frame_id2)
                    d['deceleration1'] = abs(accel1value)
                    d['deceleration2'] = abs(accel2value)
                    d['decel1_ts'] = accel1timeStr
                    d['decel2_ts'] = accel2timeStr
                    d['type'] = 0 #PET
                    d['signal_state'] = spat1
                    vinterval, vexactmatch = findPhaseDuration(vehspat, \
                            vphasetime, v[i][2])
                    d['phase_duration'] = vinterval
                    relative_time = (v[i][2] - vexactmatch.timestamp)\
                            .total_seconds()
                    d['percent_in_phase'] = relative_time*100/vinterval
                    dflist.append(d)
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
    accel1time = 0
    accel2time = 0
    accel1timeStr = None
    accel2timeStr = None
    if (accel1.empty == False):
        accel1value = accel1.iloc[accel1['accleration'].argmin()].accleration
        accel1time = accel1.iloc[accel1['accleration'].argmin()].timestamp
        #accel1time = t.strftime("%Y-%m-%d %H:%M:%S.%f")
    if (accel2.empty == False):
        accel2value = accel2.iloc[accel2['accleration'].argmin()].accleration
        accel2time = accel2.iloc[accel2['accleration'].argmin()].timestamp
        #accel2time = t.strftime("%Y-%m-%d %H:%M:%S.%f")
    if accel1time != 0:
        accel1timeStr = accel1time.strftime("%Y-%m-%d %H:%M:%S.%f")
    if accel2time != 0:
        accel2timeStr = accel2time.strftime("%Y-%m-%d %H:%M:%S.%f")
    return accel1value, accel2value, accel1time, accel2time, accel1timeStr, accel2timeStr

def getMedian(df, median_width):
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

def getCPIs(df, gp, cpi_dict, mdrac):
    for i, row in df.iterrows():
        if (row.decel1_ts != None and row.deceleration1 > mdrac[row['class1']]):
            dfobj1 = gp.get_group(row['unique_ID1'])
            index = getTimeDiffIndex(dfobj1, row.decel1_ts)
            cpi_dict[row.unique_ID1]['event'][index - 1] = 1
        if (row.decel2_ts != None and row.deceleration2 > mdrac[row['class2']]):
            dfobj2 = gp.get_group(row['unique_ID2'])
            index = getTimeDiffIndex(dfobj2, row.decel2_ts)
            cpi_dict[row.unique_ID2]['event'][index - 1] = 1
    cpi1 = [np.sum(cpi_dict[uid]['td']*cpi_dict[uid]['event']/cpi_dict[uid]['total']) for uid in df.unique_ID1]
    cpi2 = [np.sum(cpi_dict[uid]['td']*cpi_dict[uid]['event']/cpi_dict[uid]['total']) for uid in df.unique_ID2]
    return cpi1, cpi2

def do_main(body):
    (intersection_id,camera_id,start_time,end_time,nc)=json.loads(body)
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
#     threshold = 11
    
    ################################################ GET TRACKS #########################################################
    #print("Querying Tracks")
    mydb = pymysql.connect(host='maltlab.cise.ufl.edu', user='root', password='maltserver', database='testdb')
    spat_start = pd.to_datetime(start_time) - datetime.timedelta(minutes=30)
    spat_end = pd.to_datetime(end_time) + datetime.timedelta(minutes=30)
    spat_query  = "SELECT * FROM OnlineSPaT where intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,spat_start.strftime("%Y-%m-%d %H:%M:%S"),spat_end.strftime("%Y-%m-%d %H:%M:%S"))

    spatdf = pd.read_sql(spat_query, con=mydb)
    if (spatdf.empty == True):
        print ("No ATSPM signal data available")
        return
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

    sql_query  = "SELECT * FROM RealDisplayInfo where timestamp >= '{}' AND timestamp <= '{}' AND intersection_id = {} AND camera_id = {}".format(start_time,end_time,intersection_id, camera_id)
    tracks = pd.read_sql(sql_query, con=mydb)
    tracks['timestamp'] = pd.to_datetime(tracks['timestamp'])
    #print(f"Done loading /'tracks/'. No of tracks are {tracks.shape[0]}")
    
    sql_query  = "SELECT * FROM RealTrackProperties where timestamp >= '{}' AND timestamp <= '{}' AND intersection_id = {} AND camera_id = {} and isAnomalous = 0".format(start_time,end_time,intersection_id, camera_id)
    track_p = pd.read_sql(sql_query, con=mydb)
    track_p['timestamp'] = pd.to_datetime(track_p['timestamp'])
    #print(f"Done loading /'track_p/'. No of tracks are {track_p.shape[0]}")
    
    sql_query  = "SELECT * FROM TracksReal where start_timestamp >= '{}' AND start_timestamp <= '{}' AND intersection_id = {} AND camera_id = {}".format(start_time,end_time,intersection_id, camera_id)
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

    temp = tracks1.join(track_p[['unique_ID', 'phase', 'cluster', 'redJump','lanechange','nearmiss', 'city']].set_index('unique_ID'), on='unique_ID', how='inner')
    if (temp.empty == True):
        return

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
        time_diff = np.asarray(rows.td.to_list()[1:])
        event_list = np.zeros(len(time_diff))
        cpi_dict[uid] = {'cpi': 0, 'total': total_time, 'td': time_diff, 'event': event_list}
    #print("Computing pets")
    mdrac = {'car': 50, 'truck': 50, 'semi': 2,'bus': 5, 'pedestrian': 100, 'motorbike': 75, 'vehicle' : 50, 'vehicle': 50}
    petlist = compute_gaps(temp, mydb, flat_list, vehspat, vphasetime, \
        conflicting_phases, ped_conflicting_phases)
    ############################################## COMPUTE TTCs ###########################################################
    #processed_list = Parallel(n_jobs=num_cores)(delayed(TTI)(l, intersections, threshold) for _,l in cgpby)
    # Process tracks
    num_cores = 10
    processed_list = Parallel(n_jobs=num_cores)(delayed(TTI)(l, gp, vehspat, \
            vphasetime, conflicting_phases, ped_conflicting_phases) \
            for _,l in cgpby)
    count = 0
    #for indexname, rows in cgpby:
        #count = count + 1
        #TTI(rows, gp, vehspat, vphasetime, conflicting_phases, ped_conflicting_phases)
    cflat_list = [item for sublist in processed_list for item in sublist]
    if (len(cflat_list) > 0):
        cgaps = pd.DataFrame(cflat_list)
        cpi1, cpi2 = getCPIs(cgaps, gp, cpi_dict, mdrac)
        cgaps['cpi1'] = cpi1
        cgaps['cpi2'] = cpi2
        conflictsgpby = cgaps.groupby(['unique_ID1', 'unique_ID2'])
        unique_conflicts = pd.DataFrame(columns=cgaps.columns)
        for index, rows in conflictsgpby:
            unique_conflicts = unique_conflicts.append(rows[rows.time == rows.time.min()])
        count2 = 1
        uid2_conflicts = unique_conflicts
#        conflictsgpby = unique_conflicts.groupby(['unique_ID1'])
#        uid1_conflicts = pd.DataFrame(columns=cgaps.columns)
#        count1 = {}
#        for index, rows in conflictsgpby:
#            uid1_conflicts = uid1_conflicts.append(rows[rows.time == rows.time.min()])
#            count1[index] = rows.shape[0]
#        uid2_conflicts = pd.DataFrame(columns=cgaps.columns)
#        conflictsgpby = unique_conflicts.groupby(['unique_ID2'])
#        count2 = []
#        for index, rows in conflictsgpby:
#            uid2_conflicts = uid2_conflicts.append(rows[rows.time == rows.time.min()])
#            c2 = rows.shape[0]
#            if (count1.get(index)):
#                c2 = c2 + count1[index]
#            count2.append(c2)
        uid2_conflicts['num_involved'] = count2
        uid2_conflicts['intersection_diagonal'] = [mbr] * uid2_conflicts.shape[0]
        uid2_conflicts['median_width'] = getMedian(uid2_conflicts, median_width)
        uid2_conflicts['total_vehicles'] = getTotalVehicles(track_p, uid2_conflicts)


    if (len(petlist) > 0):
        petdf = pd.DataFrame(petlist)
        cpi1, cpi2 = getCPIs(petdf, gp, cpi_dict, mdrac)
        #cpi1 = [np.sum(cpi_dict[uid]['td']*cpi_dict[uid]['event']/cpi_dict[uid]['total']) for uid in petdf.unique_ID1]
        #cpi2 = [np.sum(cpi_dict[uid]['td']*cpi_dict[uid]['event']/cpi_dict[uid]['total']) for uid in petdf.unique_ID2]
        petdf['cpi1'] = cpi1
        petdf['cpi2'] = cpi2
        petdf['num_involved'] = [1] * petdf.shape[0]
        petdf['intersection_diagonal'] = [mbr] * petdf.shape[0]
        petdf['median_width'] = getMedian(petdf, median_width)
        petdf['total_vehicles'] = getTotalVehicles(track_p, petdf)

    if (len(cflat_list) > 0 or len(petlist) > 0):
        mydb = pymysql.connect(host='maltlab.cise.ufl.edu', user='root', password='maltserver', database='testdb')
        cursor=mydb.cursor()
        logging.info("Inserting records to database")
        if (len(cflat_list) > 0):
            cols = "`,`".join([str(i) for i in uid2_conflicts.columns.tolist()])
            sql = "INSERT INTO `TTCTable` (`" +cols + "`) VALUES (" + "%s,"*(uid2_conflicts.shape[1]-1) + "%s)"
            uid2_conflicts.timestamp = uid2_conflicts.timestamp.astype(str)
            tuples = [tuple(x) for x in uid2_conflicts.to_numpy()]
            cursor.executemany(sql, tuples)
        if (len(petlist) > 0):
            cols = "`,`".join([str(i) for i in petdf.columns.tolist()])
            sql = "INSERT INTO `TTCTable` (`" +cols + "`) VALUES (" + "%s,"*(petdf.shape[1]-1) + "%s)"
            petdf.timestamp = petdf.timestamp.astype(str)
            tuples = [tuple(x) for x in petdf.to_numpy()]
            cursor.executemany(sql, tuples)
        mydb.commit()
        cursor.close()

def callback(ch, method, properties, body):
    print("Processing: %s" % body, flush=True)
    file_object = file_object = open('/mnt/video/timer.txt', 'a')
    start = time.time()
    do_main(body)
    file_object.write('Finished TTC computation in {}s\n\n'.format(time.time()-start))
    file_object.close()

    ch.basic_ack(delivery_tag=method.delivery_tag)

if __name__ == "__main__":
    argdf = pd.read_csv('5060.csv', header=None)
    arglist = argdf.to_numpy().tolist()
    arglist = [[5058, "8", "2021-10-09 18:49:00.400000", "2021-10-09 18:50:05", 1]]
    arglist = [[5056, "9", "2021-10-09 17:49:00.400000", "2021-10-09 17:50:05", 1]]

    for cmd in arglist:
        print (cmd)
        do_main(json.dumps(cmd))

