#!/usr/bin/env python
# coding: utf-8

# In[1]:


import docplex.cp
import pandas as pd
import numpy as np
import json
from docplex.util.environment import get_environment


# In[2]:


path = ''
while True:
    try:
        dfTeams = pd.read_csv(path+'teams.csv', sep=';')
        TEAMS = [i for i in dfTeams.index+1]
        dfConstraints = pd.read_csv(path+'constraint_detail.csv',sep=';')
        dfDates = pd.read_csv(path+'dates.csv', sep=';')
        ROUNDS = [j for j in dfDates.index+1 if j < 15]
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


# Treat df_constraints so it's easier to obtain data

# In[3]:


#Replace team name with team number
for index, row in dfTeams.iterrows():
    dfConstraints['teama_id'] = dfConstraints['teama_id'].str.replace(row['Team'],row['Short Name'])
    dfConstraints['teamb_id'] = dfConstraints['teamb_id'].str.replace(row['Team'],row['Short Name'])


# In[4]:


matchesPerTeam = 16 #Teams must play a total of 16 matches in the Fall
UBBreaks = 2 #when a team plays two home or two consecutive rounds it is said to have a break


# ## Variables of the model
# 
# Three variables are considered:
# 
# 

# In[5]:


#from docplex.mp.model import Model
#mdl=Model('NASLS')

import docplex.cp
from docplex.cp.model import *
mdl = CpoModel(name="NASL_Scheduling")

# Variables of the model
x={}
home={}
away={}
b={}

for t in TEAMS:
    for r in ROUNDS:
        x[(t,r)] = mdl.integer_var(1, len(TEAMS),name="x_{}_{}".format(t,r)) #the opponent team of i in round r 
        home[(t,r)] = mdl.binary_var(name='h_{}_{}'.format(t,r)) #1 if team t plays home in round r
        away[(t,r)] = mdl.binary_var(name='a_{}_{}'.format(t,r)) #1 if team t plays away in round r
        b[(t,r)] = mdl.binary_var(name='b_{}_{}'.format(t,r)) #1 if team t plays home/away in BOTH round r and r+1


# ##### <font color=green> Objective Function </font>
# $Min \; Z= \; \displaystyle\sum_{t\;\in\;T}^{}\sum_{r\;\in\;R}^{} b_{tr}$

# In[6]:


sm = []
for t in TEAMS:
    for r in ROUNDS:
        sm.append(b[(t,r)])
                    
mdl.add(mdl.minimize(mdl.sum(sm)))


# In[7]:


# HARD CONSTRAINTS

for t in TEAMS:
    for r in ROUNDS[:-1]:
        #If team t plays home in r and r+1, b takes the value of 1
        mdl.add(home[(t,r)] + home[(t,r+1)] <= 1+b[(t,r)]) #(1)

for t in TEAMS:
    for r in ROUNDS[:-1]:
        #If team t plays away in r and r+1, b takes the value of 1
        mdl.add(away[(t,r)] + away[(t,r+1)] <= 1+b[(t,r)]) #(2)

for t in TEAMS:
    for r in ROUNDS:
        #Team t plays home OR away in round r
        mdl.add(home[(t,r)] != away[(t,r)]) #(3)
        
for rr in range(1,len(ROUNDS)-UBBreaks+1):
    mdl.add( mdl.sum([home[(t,r)] for t in TEAMS for r in range(rr,rr+UBBreaks+1) ]) <= UBBreaks ) #(4)

for rr in range(1,len(ROUNDS)-UBBreaks+1):
    mdl.add( mdl.sum([home[(t,r)] for t in TEAMS for r in range(rr,rr+UBBreaks+1) ]) >= 1 ) #(5)

for t1 in TEAMS:
    for t2 in TEAMS:
        if(t1!=t2):
            for r in ROUNDS[:-1]:
                mdl.add(mdl.if_then(x[(t1,r)]==t2, x[(t2,r)]==t1)) #(6)

for t1 in TEAMS:
    for t2 in TEAMS:
        if(t1<t2):
            for r in ROUNDS[:-1]:
                mdl.add(mdl.if_then(x[(t1,r)]==t2, home[(t1,r)]+home[(t2,r)] == 1)) #(12)


mdl.add(x[(t,r)] != t for t in TEAMS for r in ROUNDS) #(8)


#Not Necessary for the assignment

for t in TEAMS:
    for r in ROUNDS:
        mdl.add(x[(t,r)] > 0) #(7)
        
for r in ROUNDS:
    mdl.add(mdl.sum([home[(t,r)] for t in TEAMS]) == int(len(TEAMS)/2)) #(11)

for t in TEAMS:
    for r1 in ROUNDS:
        for r2 in ROUNDS:
            if(r1<r2):
                mdl.add(mdl.if_then(x[(t,r1)]==x[(t,r2)], home[(t,r1)]+home[(t,r2)]==1)) #(13)

#Constraints (9) and (10)
for t in TEAMS:
    QS = int(len(ROUNDS)/2)
    period1 = [x[(t,r)] for r in ROUNDS[:QS+1]]
    period2 = [x[(t,r)] for r in ROUNDS[QS+1:]]
    mdl.all_diff(period1)
    mdl.all_diff(period2)
# In[8]:


def mp_solution_to_df(solution):
    solution_df = pandas.DataFrame(columns=['name', 'value'])

    for index, dvar in enumerate(solution.iter_variables()):
        solution_df.loc[index, 'name'] = dvar.to_string()
        solution_df.loc[index, 'value'] = dvar.solution_value

    return solution_df


# In[9]:


from docplex.cp.solver import solver_local

try:
    if not mdl.solve():
        print('*** Problem has no solution')
    else:
        print('* model solved as function:')
        mdl.print_solution()
        # Save the CPLEX solution as 'solution.csv' program output
        solution_df = mp_solution_to_df(mdl.solution)
        outputs['solution'] = mdl.solution_df
except solver_local.LocalSolverException:
    print("Use Cloud Solver")
    mdl.export_model("data/Model.txt")


# In[ ]:





# In[ ]:





# In[ ]:




