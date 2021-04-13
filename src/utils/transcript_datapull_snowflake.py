#!/usr/bin/env python
# coding: utf-8

# #### Code to get the transcript table in a data frame from snowflake
# 

# In[ ]:


## Install the required packages (ONCE)
pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v2.4.1/tested_requirements/requirements_36.reqs
pip install --user snowflake-connector-python[pandas]


# In[3]:


import pandas as pd
import snowflake.connector 
import sqlalchemy






# MAKING CONNECTION
conn = snowflake.connector.connect(
  user=SECRET,
  password=SECRET,
  account=SECRET,
  warehouse=SECRET,
#  database=DATABASE,
#  schema=SCHEMA
  session_parameters={
    'QUERY_TAG': 'Transcript Data Pull'
  }
)





# MAKING CURSOR
cur = conn.cursor()
cur.execute("SELECT * FROM ""ANALYTICS_DB"".""INFO_WARRIORS"".""TRANSCRIPT""")



# IGNORE PLEASE
#cur.execute("SELECT DISTINCT org.NAME AS Group_Name,CONCAT('C', org.ID) AS ClientID,CARRIER,BENEFIT_TYPE, MONTHLY_TOTAL_PREMIUM FROM analytics.sch_analytics.fact_aggregate_carrier_client_products cp JOIN ANALYTICS.SCH_ANALYTICS.DIM_EMPLOYER_GROUPS deg ON cp.EMPLOYER_GROUP_ID=deg.ID JOIN benefits_raw.sch_benefits.organizations_s3 org ON org.NAME=deg.NAME")
#cur.execute("SELECT DISTINCT org.NAME AS Group_Name,CONCAT('C', org.ID) AS ClientID,CARRIER,BENEFIT_TYPE, MONTHLY_TOTAL_PREMIUM FROM analytics.sch_analytics.fact_aggregate_carrier_client_products cp JOIN ANALYTICS.SCH_ANALYTICS.DIM_EMPLOYER_GROUPS deg ON cp.EMPLOYER_GROUP_ID=deg.ID JOIN benefits_raw.sch_benefits.organizations_s3 org ON org.NAME=deg.NAME")
#cur.execute("SELECT DISTINCT org.NAME AS Group_Name FROM analytics.sch_analytics.fact_aggregate_carrier_client_products cp JOIN ANALYTICS.SCH_ANALYTICS.DIM_EMPLOYER_GROUPS deg ON cp.EMPLOYER_GROUP_ID=deg.ID JOIN benefits_raw.sch_benefits.organizations_s3 org ON org.NAME=deg.NAME")
#cur.execute("SELECT DISTINCT CARRIER,BENEFIT_TYPE FROM analytics.sch_analytics.fact_aggregate_carrier_client_products cp JOIN ANALYTICS.SCH_ANALYTICS.DIM_EMPLOYER_GROUPS deg ON cp.EMPLOYER_GROUP_ID=deg.ID JOIN benefits_raw.sch_benefits.organizations_s3 org ON org.NAME=deg.NAME")




# EXPORTING DATA to a pandas dataframe
transcript_df = cur.fetch_pandas_all()

