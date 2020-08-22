import pandas as pd
import numpy as np
import tensorflow as tf
import functools
from utils import preprocessing as utils
import matplotlib.pyplot as plt
from pandas.plotting import table


'''

DATA FORMAT
- Dates: YEAR-MONTH-DAY

'''

'''

HOW TO VALUATE STARTUPS

0) delete unnecessary parameters
1) get range of funds for each type of financing, assign values 
2) get range  of interval between funding rounds, assign values
3) get range of participants, assign values
4) calculate 30 > x < 70
EXTRA ANALYSIS:
3) get range degrees, assign values


'''

# normalize columns and join active startups with closed startups
''' DID IT
active_startups = pd.read_csv("../datasets/active_startups.csv")
active_startups = active_startups[utils.update_columns(active_startups)]
closed_startups = pd.read_csv("../datasets/closed_startups.csv")
closed_startups = closed_startups[utils.update_columns(closed_startups)]


# concatenate both dataframes keeping the common index (columns)
active_closed_startups = pd.concat([active_startups,closed_startups], ignore_index=True)

active_closed_startups.to_csv("../datasets/active_closed_startups.csv")


normalized_startups = pd.read_csv("../datasets/normalized_active_closed_startups.csv")

# drop duplicates to get the normalized data
normalized_startups.drop_duplicates(subset='object_id',inplace=True)

normalized_startups.to_csv("../datasets/normalized_startups.csv")
'''

startups = pd.read_csv("../datasets/normalized_startups.csv")


'''

Meaningful statistics 

'''

# get average of funding_rounds by status
average_fund_rounds_by_status = startups.groupby('status')['funding_rounds'].mean()

ax = average_fund_rounds_by_status.plot(kind='bar',title='Average of funding rounds by status')
ax.set_ylabel("Number of funding rounds")
ax.set_xlabel("Startup status")
plt.setp(ax.get_xticklabels(),rotation=45)
fig = ax.get_figure()
fig.tight_layout()

fig.savefig('../output_graphics/ avg_fund_rounds_by_status.png')

