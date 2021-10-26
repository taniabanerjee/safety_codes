import sys
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pymysql

import math 

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

intersection_id = 2175
start_time='2021-10-09 06:00:00'
end_time='2021-10-09 19:00:00'
#intersection_id = 5060
#start_time='2021-10-09 08:00:00'
#end_time='2021-10-15 19:00:00'
mydb = pymysql.connect(host=dboptions['host'], user=dboptions['user'], passwd=dboptions['passwd'], db=dboptions['testdb'], port=int(dboptions['port']))

sql_query = "SELECT * FROM TTCTable"
ttcdf = pd.read_sql(sql_query, con=mydb)
ttcdf.to_csv('TTCTable.csv', index=False)

sql_query  = "SELECT * FROM PETtable where intersection_id = '{}'  AND timestamp2 >= '{}' AND timestamp2 <= '{}'".format(intersection_id,start_time,end_time)

spat_query  = "SELECT * FROM OnlineSPaT where intersection_id = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(intersection_id,start_time,end_time)

spatdf = pd.read_sql(spat_query, con=mydb)
vehspat = spatdf[spatdf['type']==1]
pedspat = spatdf[spatdf['type']==0]

pets_orig = pd.read_sql(sql_query, con=mydb)

pets_orig['max_speed'] = pets_orig[["speed1", "speed2"]].max(axis=1)
pets = pets_orig[pets_orig.max_speed < 60]

#plt.scatter(pets['pet'], pets['max_speed'])
#plt.xlabel("PET")
#plt.ylabel("Speed (Miles per Hour)")
#plt.show()

vphasetime = vehspat.timestamp.diff().dt.total_seconds()
pedphasetime = pedspat.timestamp.diff().dt.total_seconds()

pets['inphase'] = stratify_by_phase(vehspat, pedspat, vphasetime, pedphasetime, pets, 'timestamp2')

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
unique_conflicts = unique_conflicts[(unique_conflicts.speed1 < 60) & (unique_conflicts.speed2 < 60)]
unique_conflicts['inphase'] = stratify_by_phase(vehspat, pedspat, vphasetime, pedphasetime, unique_conflicts, 'timestamp')

plt.scatter(unique_conflicts['ttc'], unique_conflicts['max_speed'])
plt.xlabel("Time to Collision")
plt.ylabel("Speed (Miles per Hour")
plt.show()

gbp = unique_conflicts.groupby('phase1')
for index, rows in gbp:
    rows.sort_values(by='max_speed', inplace=True)
    rows.to_csv('example.csv')
    #plt.scatter(rows['ttc'], rows['max_speed'])
    #plt.xlabel("Time to Collision for Phase {}".format(index))
    #plt.ylabel("Speed (Miles per Hour")
    #plt.show()


###########################################################
#################SORTING THE TABLES######################## 
###########################################################

#The first step is to compute a line passing through a random point ('ttc', 'max_speed') with slope = 1 (45 degrees)
#Using formula: y-y1 = m(x-x1) where m = 1, x1 = 'ttc', y1 = 'max_speed'
#We store the line using the format AX+BY+C = 0 (A = m = 1, B = -1, C = y1-mx1 = y1-x1)

#Returns a dictionary of A, B and C as per line formula above
def get_line_formula(tableRow):
    line={}
    line['A'] = 1
    line['B'] = -1
    line['C'] = float(tableRow['norm_max_speed']-tableRow['norm_ttc'])
    return line

#The next step is to compute the position of a point relative to a line (above, below or on the line)
#The condition for (x1, y1) to be above AX+BY+C=0: 
# 1) Ax1+By1+C>0  and b>0  2) Ax1+By1+C<0 and b<0
#The condition for (x1, y1) to be below AX+BY+C=0: 
# 1) Ax1+By1+C<0  and b>0  2) Ax1+By1+C>0 and b<0

#line: the reference line based on which the position is computed
#tableRow: containig the row (point) for which the position should be computed
def position_wrt_line(line, tableRow):
    b = line['B']
    evl = float(line['A']*tableRow['norm_ttc']+line['B']*tableRow['norm_max_speed']+line['C'])
    if evl == 0.:
        return 0
    if b > 0:
        return int(evl/abs(evl))
    else:
        return -1*int(evl/abs(evl))

#The thrid step is to compute the distance from a point to a line. Here is the formula:
#the distance from a point (x1, y1) to the  line Ax + By + C = 0 is given by 
#|AX1+BY1+C|/sqrt(A^2+B^2)

#line: the reference line based on which the position is computed
#tableRow: containig the row (point) for which the position should be computed
def get_distance(line, tableRow):
    nom = float(abs(line['A']*tableRow['norm_ttc']+line['B']*tableRow['norm_max_speed']+line['C']))
    denom = float(math.sqrt(line['A']**2 + line['B']**2))
    return position_wrt_line(line, tableRow)*(nom/denom)

def normalize_columns(table, column_list):
    for column_name in column_list:
        max_value = table[column_name].max()
        min_value = table[column_name].min()
        table['norm_{}'.format(column_name)] = (table[column_name] - min_value) / (max_value - min_value)
    return table

def sort_table(input_table):
    #first, the ttc and max_speed columns should be normalized: 
    table = normalize_columns(input_table, ['ttc', 'max_speed'])
    table.to_csv('normalized.csv')
    line = get_line_formula(table.iloc[[0]])
    table['tmp_col'] = table.apply(lambda row : -1*get_distance(line, row), axis = 1) #-1 is added to sort in descending order
    table.sort_values(by=['tmp_col', 'max_speed'], inplace=True)
    #print(table[['ttc', 'max_speed', 'tmp_col']])
    return table.drop(columns=['tmp_col', 'norm_ttc', 'norm_max_speed'])

#For debugging only
#i = 0
#for index, rows in gbp:
#    table = sort_table(rows)
#    table.to_csv('test_sort{}.csv'.format(i))   
#    i+=1

#For debugging only
table = sort_table(unique_conflicts)
table.to_csv('sorted_uniqu.csv')


