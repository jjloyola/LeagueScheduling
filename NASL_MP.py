#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
from docplex.mp.model import Model
from docplex.util.environment import get_environment


# In[2]:


path = ''
while True:
    try:
        dfTeams = pd.read_csv(path+'teams.csv', sep=';')
        dfConstraints = pd.read_csv(path+'constraint_detail.csv',sep=';')
        dfDates = pd.read_csv(path+'dates.csv', sep=';')
        break
    except IOError:
        path = 'data/'
    

dfDistances = pd.DataFrame(data=0.0,
                    index = dfTeams['Short Name'],
                    columns = dfTeams['Short Name'])

with open(path+'distances.json') as dist_file:  
    data = json.load(dist_file)

    for team1 in data['travel_distances']:
        for team2 in team1['distances']:
            if team1['short_name'] != team2['short_name']:
                dfDistances[team1['short_name']][team2['short_name']] = team2['distance']

#TEMP PARAMS
n = 8
TEAMS = {}
for i in dfTeams.index:
    TEAMS[dfTeams.loc[i,'Short Name']]= i+1
    
nbRounds = n-1;
ROUNDS = {}
ROUNDS_Y = []
for j in dfDates.index:
    ROUNDS[dfDates.loc[j,'Date']]= j+1
    ROUNDS_Y.append(j+1)

ROUNDS_Y=ROUNDS_Y[:-1]
UBBreaks = 2 #when a team plays two home or two consecutive rounds it is said to have a break, upper bound will be 2 in-a-row matches


# Treat df_constraints so it's easier to obtain data

# In[3]:


#Replace team name with team short name
for index, row in dfTeams.iterrows():
    dfConstraints['teama_id'] = dfConstraints['teama_id'].str.replace(row['Team'],row['Short Name'])
    dfConstraints['teamb_id'] = dfConstraints['teamb_id'].str.replace(row['Team'],row['Short Name'])


# ## Variables of the model

# In[4]:


mdl = Model('NASL_Scheduling')

# Variables of the model
home={}
away={}

for t in TEAMS.values():
    for r in ROUNDS_Y:
        home[(t,r)] = mdl.binary_var(name='h_{}_{}'.format(t,r)) #1 if team t plays home in round r and r+1
        away[(t,r)] = mdl.binary_var(name='a_{}_{}'.format(t,r)) #1 if team t plays away in round r and r+1


plays = {}
for r in ROUNDS.values():
    for t1 in TEAMS.values():
        for t2 in TEAMS.values():
            plays[(t1, t2, r)] = mdl.binary_var(name="x_{}_{}_{}".format(t1, t2, r))


# ### <font color=green> Objective Function </font>
# $Min \; Z= \; \displaystyle\sum_{t \in T}^{}\sum_{r \in R-1}^{} h_{tr} + a_{tr}$

# In[5]:


sm = []
for t in TEAMS.values():
    for r in ROUNDS_Y:
        sm.append(home[(t,r)]+away[(t,r)])
                    
mdl.minimize(mdl.sum(sm))


# * All teams need to play 16 matches
# * Every possible matchup has to be played between 2 and 3 times. 
# * At least one match against every opponent has to be played as a home match, and at least one has to be played as an away game.
# * 25 rounds
# * As many matches as possible on weekends, but for TV purposes it would be ideal to have one match on every Wednesday
# 

# In[6]:


#(1) Each team must play every other team at least 2 times, and at most 3 times
for t1 in TEAMS.values():
    for t2 in TEAMS.values():
        if t1!=t2:
            mdl.add_constraint(mdl.sum([plays[(t1,t2,r)] + plays[(t2,t1,r)] for r in ROUNDS.values()]) >=2) #(1)
            mdl.add_constraint(mdl.sum([plays[(t1,t2,r)] + plays[(t2,t1,r)] for r in ROUNDS.values()]) <= 3) #(1)
            
#x_1_2_1 + x_2_1_1 + x_1_2_2 + x_2_1_2 + x_1_2_3 + x_2_1_3 + x_1_2_4 + x_2_1_4 + x_1_2_5 + x_2_1_5 + x_1_2_6 + x_2_1_6 + x_1_2_7 + x_2_1_7 = 1
#x_1_3_1 + x_3_1_1 + x_1_3_2 + x_3_1_2 + x_1_3_3 + x_3_1_3 + x_1_3_4 + x_3_1_4 + x_1_3_5 + x_3_1_5 + x_1_3_6 + x_3_1_6 + x_1_3_7 + x_3_1_7 = 1
#x_1_4_1 + x_4_1_1 + x_1_4_2 + x_4_1_2 + x_1_4_3 + x_4_1_3 + x_1_4_4 + x_4_1_4 + x_1_4_5 + x_4_1_5 + x_1_4_6 + x_4_1_6 + x_1_4_7 + x_4_1_7 = 1



#(2) In a round r and against any opponent team i, team j plays either home or away, not both.
for t1 in TEAMS.values():
    for r in ROUNDS.values():
        sumAux = []
        for t2 in TEAMS.values():
            if t1!=t2:
                sumAux.append(plays[(t1,t2,r)] + plays[(t2,t1,r)])
        mdl.add_constraint(mdl.sum(sumAux) <= 1) #(2)

#x_1_2_1 + x_1_3_1 + x_1_4_1 + x_1_5_1 + x_1_6_1 + x_1_7_1 + x_1_8_1 + x_2_1_1 + x_3_1_1 + x_4_1_1 + x_5_1_1 + x_6_1_1 + x_7_1_1 + x_8_1_1 = 1
#Means that un round 1, no matter against whom, team 1 plays either home or away.
#x_1_2_2 + x_1_3_2 + x_1_4_2 + x_1_5_2 + x_1_6_2 + x_1_7_2 + x_1_8_2 + x_2_1_2 + x_3_1_2 + x_4_1_2 + x_5_1_2 + x_6_1_2 + x_7_1_2 + x_8_1_2 = 1
#x_1_2_3 + x_1_3_3 + x_1_4_3 + x_1_5_3 + x_1_6_3 + x_1_7_3 + x_1_8_3 + x_2_1_3 + x_3_1_3 + x_4_1_3 + x_5_1_3 + x_6_1_3 + x_7_1_3 + x_8_1_3 = 1
        
        
#(3) No team i play itself in any round r.
for t in TEAMS.values():
    for r in ROUNDS.values():
        mdl.add_constraint(plays[(t,t,r)]==0)


# In[7]:


#Each team plays 8 matches at home and 8 matches away
for t in TEAMS.values():
    mdl.add_constraint(mdl.sum([plays[(t,t2,r)] for t2 in TEAMS.values() for r in ROUNDS.values() if t!=t2]) == 8)
    mdl.add_constraint(mdl.sum([plays[(t2,t,r)] for t2 in TEAMS.values() for r in ROUNDS.values() if t!=t2]) == 8)


# In[8]:


#(6) Every team has a maximum of two consecutive home matches.
for t in TEAMS.values():
    mdl.add_constraint(mdl.sum([home[(t,r)] for r in ROUNDS_Y]) <= UBBreaks-1) #(6)

#(7) Every team has a maximum of two consecutive away matches.
for t in TEAMS.values():
    mdl.add_constraint(mdl.sum([away[(t,r)] for r in ROUNDS_Y]) <= UBBreaks-1) #(7)


# In[9]:


#(8) Making the h variable dependent on the x variable.
for t1 in TEAMS.values():
    for r in ROUNDS_Y:
        mdl.add_constraint(mdl.sum([plays[(t2,t1,r)]+plays[(t2,t1,r+1)] for t2 in TEAMS.values() if t1!=t2]) 
                           <= 1+home[(t1,r)]) #(8)


#(9) Making the a variable dependent on the x variable.
for t1 in TEAMS.values():
    for r in ROUNDS_Y:
        mdl.add_constraint(mdl.sum([plays[(t2,t1,r)]+plays[(t2,t1,r+1)] for t2 in TEAMS.values() if t1!=t2]) 
                           <= 1+away[(t1,r)]) #(9)


# In[10]:


def getRoundID(dateStr):
    if str(dateStr) in ROUNDS.keys():
        return ROUNDS[dateStr]
    else:
        return -1

def getTeamIDList(shortNameStr):
    strList = shortNameStr.replace(' ','').split(';')
    teamIDList = []
    for shortName in strList:
        if shortName in TEAMS.keys():
            teamIDList.append(TEAMS[shortName])
    return teamIDList

def getTeamID(shortName):
    if str(shortName) in TEAMS.keys():
        return TEAMS[shortName]
    else:
        return -1
    

def getRowsByConstraintDesc(constraintDescriptionStr):
    newDF = dfConstraints.loc[dfConstraints['constraint_description'] == constraintDescriptionStr]
    return newDF

def getRowsByCtType(dfSource, ctTypeStr):
    return dfSource.loc[dfConstraints['type'] == ctTypeStr]

def getConstraintInfo(row):
    ctDict = {}
    ctDict['roundNum_begin'] = getRoundID(row['begin_date'])
    ctDict['roundNum_end'] = getRoundID(row['end_date'])
    ctDict['hostID'] = getTeamID(row['teama_id'])
    ctDict['visitorID'] = getTeamID(row['teamb_id'])
    
    return ctDict


# In[11]:


# Must Host (DerivedConstraint::HomeRequest)
# SFD must host on one of the specified dates
for index,row in dfConstraints.loc[dfConstraints['constraint_description'] == 'Must Host'].iterrows():
    ctDict = getConstraintInfo(row)
    r = ctDict['roundNum_begin']
    while r <= ctDict['roundNum_end']:
        mdl.add_constraint(mdl.sum([plays[(ctDict['hostID'],visitor,r)] for visitor in TEAMS.values()]) == 1,
                           ctname='Must_Host_{}_R{}'.format(row['teama_id'], r))
        r=r+1

# Home Blackouts (DerivedConstraint::HomeBlackout)
for index,row in getRowsByConstraintDesc('Home Blackouts').iterrows():
    ctDict = getConstraintInfo(row)
    r = ctDict['roundNum_begin']
    while r <= ctDict['roundNum_end']:
        mdl.add_constraint(mdl.sum([plays[(ctDict['hostID'],visitor,r)] for visitor in TEAMS.values()]) == 0,
                           ctname='Home_Blackout_{}_R{}'.format(row['teama_id'],r))
        r=r+1


# In[12]:


# JAX Post Midweek Matchup Blackouts (DerivedConstraint::MatchupBlackout)
for index,row in getRowsByConstraintDesc('JAX Post Midweek Matchup Blackouts').iterrows():
    ctDict = getConstraintInfo(row)
    
    r = ctDict['roundNum_begin']
    while r <= ctDict['roundNum_end']:
        mdl.add_constraint(plays[(ctDict['hostID'],ctDict['visitorID'],r)] == 0,
                           ctname='Matchup_Blackout_{}_{}_R{}'.format(row['teama_id'], row['teamb_id'], r))
        r=r+1


# In[13]:


# Jacksonville Midweek Home Match Request (DerivedConstraint::MatchRequest)
for index,row in getRowsByConstraintDesc('Jacksonville Midweek Home Match Request').iterrows():
    ctDict = getConstraintInfo(row)
    visitorList = getTeamIDList(row['teamb_id'])

    r = ctDict['roundNum_begin']
    while r <= ctDict['roundNum_end']:
        matches = []
        for v in visitorList:
            matches.append(plays[(ctDict['hostID'], v, r)])
        mdl.add_constraint(mdl.sum(matches) == 1,
                           ctname='Home_Match_Request_{}_R{}'.format(row['teama_id'], r))
        r=r+1


# In[14]:


# Jacksonville Midweek Must Host (DerivedConstraint::HomeRequest)
# JAX plays home on 8/16/2017 OR JAX plays home 10/25/2017
for index,row in getRowsByConstraintDesc('Jacksonville Midweek Must Host').iterrows():
    ctDict = getConstraintInfo(row)
    r = ctDict['roundNum_begin']
    while r <= ctDict['roundNum_end']:
        mdl.add_constraint(mdl.sum([plays[(ctDict['hostID'],visitor,r)] for visitor in TEAMS.values()]) == 1,
                           ctname='Must_Host_{}_R{}'.format(row['teama_id'],r))
        r=r+1


# In[15]:


# First Round Request (DerivedConstraint::HomeRequest)
# Teams specified whether with whom they want to play on the first match.

for index,row in getRowsByConstraintDesc('First Round Request').iterrows():
    ctDict = getConstraintInfo(row)
    # Two possible meanings: for the multiple row constraint
    
    r = ctDict['roundNum_begin']
    while r <= ctDict['roundNum_end']:
        if str(row['type']).find('HomeRequest') != -1:
            # MIA and NYC and JAX and CAR will play home on 7/29/2017 <--- using this
            # MIA or NYC or JAX or CAR play home on 7/29/2017
            mdl.add_constraint(mdl.sum( [plays[(ctDict['hostID'], visitor, r)] for visitor in TEAMS.values()] ) == 1, 
                               ctname='First_Round_Home_{}_R{}'.format(row['teama_id'], r))
        elif str(row['type']).find('AwayRequest') != -1:
            # SFD and PRFC and IND play away on 7/29/2017 <-- using this
            # SFD or PRFC or IND play away on 7/29/2017
            mdl.add_constraint(mdl.sum([plays[(host, ctDict['visitorID'], r)] for host in TEAMS.values() if ctDict['visitorID']!= host]) == 1,
                               ctname='First_Round_Away_{}_R{}'.format(row['teamb_id'], r))
        r=r+1


# In[16]:


def mp_solution_to_df(solution):
    solution_df = pd.DataFrame(columns=['name', 'value'])

    for index, dvar in enumerate(solution.iter_variables()):
        solution_df.loc[index, 'name'] = dvar.to_string()
        solution_df.loc[index, 'value'] = dvar.solution_value

    print(solution_df)    
    
    return solution_df


# In[19]:


#mdl.parameters.timelimit=10

try:
    if not mdl.solve():
        print('*** Problem has no solution')
    else:
        print('* model solved as function:')
        # Save the CPLEX solution as 'solution.csv' program output
        solution_df = mp_solution_to_df(mdl.solution)
        get_environment().store_solution(solution_df)
        print("\n\n---------------------SOLUTION--------------------\n\n")
        print(solution_df)
except Exception:
    print("Use Cloud Solver.", Exception.args)
    if(path!=""):
        mdl.export_as_lp(basename="MP_Model", path=path)

