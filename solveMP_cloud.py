#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
This example demonstrate how to run a python model on DOcplexcloud solve
service.

@author: kong
'''
from docloud.job import JobClient
from docplex.mp.context import Context


# In[2]:


# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2017
# --------------------------------------------------------------------------



if __name__ == '__main__':
    '''DOcplexcloud credentials can be specified with url and api_key in the
    code block below.

    Alternatively, Context.make_default_context() searches the PYTHONPATH for
    the following files:

        * cplex_config.py
        * cplex_config_<hostname>.py
        * docloud_config.py (must only contain context.solver.docloud configuration)

    These files contain the credentials and other properties. For example,
    something similar to::

       context.solver.docloud.url = 'https://docloud.service.com/job_manager/rest/v1'
       context.solver.docloud.key = 'example api_key'
    '''
    url = 'https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1'
    key = 'example api_key'

#    if url is None or key is None:
        # create a default context and use credentials defined in there.
#        context = Context.make_default_context()
#        url = context.solver.docloud.url
#        key = context.solver.docloud.key

    client = JobClient(url=url, api_key=key)
    resp = client.execute(input=['NASL_MP.py',
                                 'data/teams.csv',
                                 'data/constraint_detail.csv',
                                 'data/dates.csv',
                                 'data/distances.json'],
                          output='solution.json', 
                          waittime=60,
                          load_solution=True,
                          log='logs.txt')


# In[ ]:




