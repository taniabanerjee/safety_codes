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

intersection_id = 2175
start_time='2021-08-24 08:00:00'
end_time='2021-08-24 19:00:00'
mydb = pymysql.connect(host=dboptions['host'], user=dboptions['user'], passwd=dboptions['passwd'], db=dboptions['testdb'], port=int(dboptions['port']))

sql_query  = "SELECT * FROM PETtable where intersection_id = '{}'  AND timestamp2 >= '{}' AND timestamp2 <= '{}'".format(intersection_id,start_time,end_time)

pets = pd.read_sql(sql_query, con=mydb)

pets['max_speed'] = pets[["speed1", "speed2"]].max(axis=1)

plt.scatter(pets['pet'], pets['max_speed'])
plt.xlabel("PET")
plt.ylabel("Speed (Miles per Hour)")
plt.show()

sql_query  = "SELECT * FROM TTCTable where intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,start_time,end_time)
sql_query2  = "SELECT * FROM RealTrackProperties where intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{} and (isAnomalous != 1 or lanechane = 1)'".format(intersection_id,start_time,end_time)

conflicts = pd.read_sql(sql_query, con=mydb)
tracks = pd.read_sql(sql_query2, con=mydb)

conflicts['timestamp'] = pd.to_datetime(conflicts['timestamp'])
tracks['timestamp'] = pd.to_datetime(tracks['timestamp'])

congpby = conflicts.groupby(['unique_ID1', 'unique_ID2'])
unique_conflicts = pd.DataFrame(columns=conflicts.columns)
for index, rows in congpby:
    unique_conflicts = unique_conflicts.append(rows[rows.ttc == rows.ttc.min()])

unique_conflicts['approach1'] = unique_conflicts['cluster1'].astype(str).str[0]
unique_conflicts['approach2'] = unique_conflicts['cluster2'].astype(str).str[0]
unique_conflicts['max_speed'] = unique_conflicts[["speed1", "speed2"]].max(axis=1)
unique_conflicts = unique_conflicts[unique_conflicts.cluster1 != unique_conflicts.cluster2]
unique_conflicts = unique_conflicts[unique_conflicts.approach1 != unique_conflicts.approach2]

plt.scatter(unique_conflicts['ttc'], unique_conflicts['max_speed'])
plt.xlabel("Time to Collision")
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

print ('Minimum:', np.min(unique_conflicts['ttc']))
print ('Maximum:', np.max(unique_conflicts['ttc']))
print ('Average:', np.mean(unique_conflicts['ttc']))
print ('Median:', np.median(unique_conflicts['ttc']))

plt.bar(range(8, 19), agg_by_hour['timestamp'])
plt.xticks(range(8, 19), labels=range(8, 19))
plt.xlabel("Hour of day")
plt.ylabel("Number of events")
plt.title("Nearmiss events by the hour")
plt.show()

plt.hist(unique_conflicts['ttc'], bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
plt.xlabel("Minimum TTC")
plt.ylabel("Counts")
plt.show()

cgpby = unique_conflicts.groupby(['cluster1', 'cluster2'])
for index, rows in cgpby:
    print (index, len(rows))

