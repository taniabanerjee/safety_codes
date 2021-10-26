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
    df_t['absspeed'] = df_t['inst_speed'].rolling(window=3).mean()
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

def TTI(df_gppy, cpi_dict, threshold=2): # , cpi_frame=None
    drac = {'car': 50, 'truck': 50, 'semi': 2,'bus': 5, 'pedestrian': 100, 'motorbike': 75, 'vehicle' : 50}
    
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
            conflict_x, conflict_y, t1, t2 = findTTC(cx1, cy1, cx2, cy2, vx1, vy1, vx2, vy2) 
            diffabs = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
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
            drac1 = (vx1**2)/(2*(vx1*cx1*t1)+0.001)  # V^2/2s
            drac2 = (vx2**2)/(2*(vx2*cx2*t2)+0.001)
            madr1 = drac[rowi['class']] # deceleration CPI FORMULA
            madr2 = drac[rowj['class']]
            cpi1, cpi2 = [int(d>m) for d, m in [(drac1, madr1), (drac2, madr2)]]
            cp1 = cpi1*rowi['td']
            for ID, t, c in [(rowi['unique_ID'],rowi['timestamp'],cpi1), (rowj['unique_ID'],rowj['timestamp'],cpi2)]:
                total = cpi_dict[ID]['total']
                c = c / total if total != 0 else 0
                cpi_dict[ID]['cpi'] = cpi_dict[ID]['cpi'] + c
            
            ## Encode Table ##
            d = {}
            d['intersection_id'] = rowi['intersection_id']
            d['camera_id'] = rowi['camera_id']
            d['timestamp'] = rowi['timestamp']
            d['frame_id'] = rowi['frame_id']
            d['conflict_x'], d['conflict_y'] = conflict_x, conflict_y
            d['unique_ID1'], d['unique_ID2'] = rowi['unique_ID'], rowj['unique_ID']
            d['class1'], d['class2'] = rowi['class'], rowj['class']
            d['phase1'],   d['phase2'] = rowi.phase, rowj.phase
            d['distance'] = diffabs
            #d['ttc'] = 1/((abs(t1-t2)+.1)*(t1*t2+1))
            d['ttc'] = (t1 + t2)/2
            d['p2v'] = p2v
            d['city'] = rowi['city']
            d['cluster1'], d['cluster2'] = rowi['cluster'], rowj['cluster']
            d['speed1'] = np.sqrt(vx1**2 + vy1**2) * (3 * 2.23694/50)
            d['speed2'] = np.sqrt(vx2**2 + vy2**2) * (3 * 2.23694/50)
            list_.append(d)           
    return list_

def do_main(body):
    (intersection_id,camera_id,start_time,end_time)=json.loads(body)
#     threshold = 11
    
    ################################################ GET TRACKS #########################################################
    #print("Querying Tracks")
    mydb = pymysql.connect(host='maltlab.cise.ufl.edu', user='root', password='maltserver', database='testdb')
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
        start_time = rows['timestamp'].to_list()[firstindex]
        end_time = rows['timestamp'].to_list()[lastindex]
        cpi_dict[uid] = {'cpi': 0, 'total': (end_time - start_time).total_seconds()}
    #print("Computing ttcs")
    ############################################## COMPUTE TTCs ###########################################################
    #processed_list = Parallel(n_jobs=num_cores)(delayed(TTI)(l, intersections, threshold) for _,l in cgpby)
    # Process tracks
    num_cores = 10
    processed_list = Parallel(n_jobs=num_cores)(delayed(TTI)(l, cpi_dict) for _,l in cgpby)
    #for indexname, rows in cgpby:
        #TTI(rows, cpi_dict)
    cflat_list = [item for sublist in processed_list for item in sublist]
    cpi_df = pd.DataFrame.from_dict(cpi_dict).T  ##### Unique column name
    update_cpi_query='update TracksReal set cpi=%s where unique_ID=%s;'
    updaterows = []
    for key in cpi_dict:
        if (cpi_dict[key]['cpi'] > 0):
            updaterows.append((str(cpi_dict[key]['cpi']), str(key)))
    if updaterows:
        cursor.executemany(updatetrackquery, updaterows)
    if (len(cflat_list) > 0):
        cgaps = pd.DataFrame(cflat_list)
        
        mydb = pymysql.connect(host='maltlab.cise.ufl.edu', user='root', password='maltserver', database='testdb')
        cursor=mydb.cursor()
        cols = "`,`".join([str(i) for i in cgaps.columns.tolist()])
        sql = "INSERT INTO `TTCTable` (`" +cols + "`) VALUES (" + "%s,"*(cgaps.shape[1]-1) + "%s)"
        logging.info("Inserting records to database")
        cgaps.timestamp = cgaps.timestamp.astype(str)
        tuples = [tuple(x) for x in cgaps.to_numpy()]
        cursor.executemany(sql, tuples)
        mydb.commit()
        cursor.close()

if __name__ == "__main__":
#    q = FifoDiskQueue("../queues/dt_ttc_ready")
#    qpop=q.pop()
#    q.close()
#    if (qpop != None):
#        arglist=qpop.decode()
#        do_main(arglist)
#    else:
#        sys.exit("No message found")

    argdf = pd.read_csv('test.csv', header=None)
    arglist = argdf.to_numpy().tolist()

    for cmd in arglist:
        print (cmd)
        do_main(json.dumps(cmd))

