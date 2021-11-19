import sys
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pymysql

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

def stratify_by_phase(vehspat, pedspat, vphasetime, pedphasetime, ttc_pet, field):
    inphase = []
    for ts in ttc_pet[field]:
        vindex = np.searchsorted(vehspat.timestamp, ts)
        vexactmatch = vehspat.iloc[vindex-1]
        pindex = np.searchsorted(pedspat.timestamp, ts)
        pexactmatch = pedspat.iloc[pindex-1]
    
        vrs = vexactmatch['hexphase']
        cycle = vexactmatch['cycle']
        try:
            vinterval = vphasetime.iloc[vindex]
        except:
            vinterval = 0
        prs = pexactmatch['hexphase']
        try:
            pinterval = pedphasetime.iloc[pindex]
        except:
            pinterval = 0
    
        relative_time = (ts-vexactmatch.timestamp).total_seconds()
        if (relative_time < vinterval*0.10):
            inphase.append(vrs + '-begin')
        elif (relative_time > vinterval*0.9):
            inphase.append(vrs + '-end')
        else:
            inphase.append(vrs + '-mid')
    
    ttc_pet['inphase'] = inphase
    
    (unique, counts) = np.unique(inphase, return_counts = True)
    print (unique)
    print (counts)
    return inphase

#intersection_id = 2175
#start_time='2021-10-09 06:00:00'
#end_time='2021-10-09 19:00:00'
intersection_id = 5060
start_time='2021-10-09 08:00:00'
end_time='2021-10-09 19:00:00'
mydb = pymysql.connect(host=dboptions['host'], user=dboptions['user'], passwd=dboptions['passwd'], db=dboptions['testdb'], port=int(dboptions['port']))

sql_query = "SELECT * FROM TTCTable where type = 1 and intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,start_time,end_time)
ttcdf = pd.read_sql(sql_query, con=mydb)
ttcdf.to_csv('TTCTable.csv', index=False)

sql_query  = "SELECT * FROM TTCTable where type = 0 and intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,start_time,end_time)

spat_query  = "SELECT * FROM OnlineSPaT where intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,start_time,end_time)

spatdf = pd.read_sql(spat_query, con=mydb)
vehspat = spatdf[spatdf['type']==1]
pedspat = spatdf[spatdf['type']==0]

pets_orig = pd.read_sql(sql_query, con=mydb)

pets_orig['max_speed'] = pets_orig[["speed1", "speed2"]].max(axis=1)
pets = pets_orig[pets_orig.max_speed < 60]


plt.scatter(pets['time'], pets['max_speed'])
plt.xlabel("PET")
plt.ylabel("Speed (Miles per Hour)")
plt.show()

vphasetime = vehspat.timestamp.diff().dt.total_seconds()
pedphasetime = pedspat.timestamp.diff().dt.total_seconds()

#pets['inphase'] = stratify_by_phase(vehspat, pedspat, vphasetime, pedphasetime, pets, 'timestamp2')

sql_query  = "SELECT * FROM TTCTable where intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,start_time,end_time)
sql_query2  = "SELECT * FROM RealTrackProperties where intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{} and (isAnomalous != 1 or lanechane = 1)'".format(intersection_id,start_time,end_time)

conflicts = pd.read_sql(sql_query, con=mydb)
tracks = pd.read_sql(sql_query2, con=mydb)

conflicts['timestamp'] = pd.to_datetime(conflicts['timestamp'])
tracks['timestamp'] = pd.to_datetime(tracks['timestamp'])

congpby = conflicts.groupby(['unique_ID1', 'unique_ID2'])
unique_conflicts = pd.DataFrame(columns=conflicts.columns)
for index, rows in congpby:
    unique_conflicts = unique_conflicts.append(rows[rows.time == rows.time.min()])

unique_conflicts['approach1'] = unique_conflicts['cluster1'].astype(str).str[0]
unique_conflicts['approach2'] = unique_conflicts['cluster2'].astype(str).str[0]
unique_conflicts['max_speed'] = unique_conflicts[["speed1", "speed2"]].max(axis=1)
unique_conflicts = unique_conflicts[unique_conflicts.cluster1 != unique_conflicts.cluster2]
unique_conflicts = unique_conflicts[unique_conflicts.approach1 != unique_conflicts.approach2]
unique_conflicts = unique_conflicts[(unique_conflicts.speed1 < 60) & (unique_conflicts.speed2 < 60)]
#unique_conflicts['inphase'] = stratify_by_phase(vehspat, pedspat, vphasetime, pedphasetime, unique_conflicts, 'timestamp')

#unique_conflicts = unique_conflicts[(unique_conflicts.distance < 170) & (unique_conflicts.speed1 > 10) & (unique_conflicts.speed2 > 10) & (unique_conflicts.deceleration1 > 0) & (unique_conflicts.deceleration2 > 0) & (unique_conflicts.time < 10) & (unique_conflicts.time > 0)]
unique_conflicts = unique_conflicts[(unique_conflicts.distance < 170) & (unique_conflicts.speed1 > 10) & (unique_conflicts.speed2 > 10) & (unique_conflicts.deceleration1 > 0) & (unique_conflicts.time > 0)]
print (unique_conflicts)

plt.scatter(unique_conflicts['time'], unique_conflicts['max_speed'])
plt.xlabel("Time to Collision")
plt.ylabel("Speed (Miles per Hour")
plt.show()

gbp = unique_conflicts.groupby('phase1')
for index, rows in gbp:
    plt.scatter(rows['time'], rows['max_speed'])
    plt.xlabel("Time to Collision for Phase {}".format(index))
    plt.ylabel("Speed (Miles per Hour")
    plt.show()

agg_by_hour = unique_conflicts.resample('H', on='timestamp').count()
tracks_by_hour = tracks.resample('H', on='timestamp').count()

plt.bar(range(8, 19), tracks_by_hour['timestamp'])
plt.xticks(range(8, 19), labels=range(8, 19))
plt.xlabel("Hour of day")
plt.ylabel("Number of trajectories")
plt.title("Total trajectories by the hour")
plt.show()

print ('Minimum:', np.min(unique_conflicts['time']))
print ('Maximum:', np.max(unique_conflicts['time']))
print ('Average:', np.mean(unique_conflicts['time']))
print ('Median:', np.median(unique_conflicts['time']))

plt.bar(range(8, 19), agg_by_hour['timestamp'])
plt.xticks(range(8, 19), labels=range(8, 19))
plt.xlabel("Hour of day")
plt.ylabel("Number of events")
plt.title("Nearmiss events by the hour")
plt.show()

plt.hist(unique_conflicts['time'], bins=np.linspace(0, 10, 101))
plt.xlabel("Minimum TTC")
plt.ylabel("Counts")
plt.show()

cgpby = unique_conflicts.groupby(['cluster1', 'cluster2'])
for index, rows in cgpby:
    print (index, len(rows))

