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
        #TEAMS = [i for i in dfTeams.index+1]
        dfConstraints = pd.read_csv(path+'constraint_detail.csv',sep=';')
        dfDates = pd.read_csv(path+'dates.csv', sep=';')
        #ROUNDS = [j for j in dfDates.index+1]
        #ROUNDS = [j for j in TEAMS[:-1]] #REMOVE
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
TEAMS = range(1,n+1)
nbRounds = n-1;
ROUNDS = range(1,nbRounds+1)
ROUNDS_Y =range(1,nbRounds)
mid = nbRounds//2
#UB = 2


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


mdl = Model('NASL_Scheduling')

# Variables of the model
#x={}
home={}
away={}
#b={}

for t in TEAMS:
    for r in ROUNDS_Y:
        home[(t,r)] = mdl.binary_var(name='h_{}_{}'.format(t,r)) #1 if team t plays home in round r and r+1
        away[(t,r)] = mdl.binary_var(name='a_{}_{}'.format(t,r)) #1 if team t plays away in round r and r+1


# In[6]:


plays = {}
for r in ROUNDS:
    for t1 in TEAMS:
        for t2 in TEAMS:
            plays[(t1, t2, r)] = mdl.binary_var(name="x_{}_{}_{}".format(t1, t2, r))


# ### <font color=green> Objective Function </font>
# $Min \; Z= \; \displaystyle\sum_{t \in T}^{}\sum_{r \in R-1}^{} h_{tr} + a_{tr}$

# In[7]:


sm = []
for t in TEAMS:
    for r in ROUNDS_Y:
        sm.append(home[(t,r)]+away[(t,r)])
                    
mdl.minimize(mdl.sum(sm))


# (1) Make sure that each team i plays every other team j once in the period of 7
# rounds.
# 
# (2) In every round r and for every team i, team j plays either home or away.
# 
# (3) No team i play itself in any round r.
# 
# (4) – (5) Each team plays at least 7 and maximum 8 matches at home.
# 
# (6) Every team has a maximum of two consecutive home matches.
# 
# (7) Every team has a maximum of two consecutive away matches.
# 
# (8) Making the h variable dependent on the x variable.
# 
# (9) Making the a variable dependent on the x variable.

# In[8]:


for t1 in TEAMS:
    for t2 in TEAMS:
        if t1!=t2:
            mdl.add_constraint(mdl.sum([plays[(t1,t2,r)] + plays[(t2,t1,r)] for r in ROUNDS]) == 1) #(1)

for t1 in TEAMS:
    for r in ROUNDS:
        sumAux = []
        for t2 in TEAMS:
            if t1!=t2:
                sumAux.append(plays[(t1,t2,r)] + plays[(t2,t1,r)])
        mdl.add_constraint(mdl.sum(sumAux) == 1) #(2)

for t in TEAMS:
    for r in ROUNDS:
        mdl.add_constraint(plays[(t,t,r)]==0)


# In[9]:


for t in TEAMS:
    mdl.add_constraint(mdl.sum([plays[(t,t2,r)] for t2 in TEAMS for r in ROUNDS if t!=t2]) <= mid+1)#(4)

for t in TEAMS:
    mdl.add_constraint(mdl.sum([plays[(t,t2,r)] for t2 in TEAMS for r in ROUNDS if t!=t2]) >= mid)#(5)


# In[10]:


for t in TEAMS:
    mdl.add_constraint(mdl.sum([home[(t,r)] for r in ROUNDS_Y]) <= UBBreaks-1) #(6)

for t in TEAMS:
    mdl.add_constraint(mdl.sum([away[(t,r)] for r in ROUNDS_Y]) <= UBBreaks-1) #(7)


# In[11]:


for t1 in TEAMS:
    for r in ROUNDS_Y:
        mdl.add_constraint(mdl.sum([plays[(t2,t1,r)]+plays[(t2,t1,r+1)] for t2 in TEAMS if t1!=t2]) 
                           <= 1+home[(t1,r)]) #(8)

for t1 in TEAMS:
    for r in ROUNDS_Y:
        mdl.add_constraint(mdl.sum([plays[(t2,t1,r)]+plays[(t2,t1,r+1)] for t2 in TEAMS if t1!=t2]) 
                           <= 1+away[(t1,r)]) #(9)


# In[12]:


#mdl.export_as_lp(basename="MP_Model", path="data/")


# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


def mp_solution_to_df(solution):
    solution_df = pd.DataFrame(columns=['name', 'value'])

    for index, dvar in enumerate(solution.iter_variables()):
        solution_df.loc[index, 'name'] = dvar.to_string()
        solution_df.loc[index, 'value'] = dvar.solution_value

    return solution_df


# In[16]:


#mdl.parameters.timelimit=10

try:
    if not mdl.solve():
        print('*** Problem has no solution')
    else:
        print('* model solved as function:')
        # Save the CPLEX solution as 'solution.csv' program output
        solution_df = mp_solution_to_df(mdl.solution)
        #get_environment().store_solution(solution_df)
        print("\n\n---------------------SOLUTION--------------------\n\n")
        mdl.print_solution()
except Exception:
    print("Use Cloud Solver")
    mdl.export_as_lp(basename="MP_Model", path="data/")


# In[ ]:





# In[ ]:




