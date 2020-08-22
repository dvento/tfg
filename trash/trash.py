import pandas as pd
import numpy as np
import tensorflow as tf
import functools
from utils import preprocessing as utils


'''

THIS DOCUMENT IS USED TO SAVE UNUSED METHODS AND CLASSES THAT MIGHT BE 
USEFUL IN THE FUTURE

'''

'''
ANOTHER FILE
traditional.py  code ABOVE

'''


'''
last_id = 1
id = 0

for i in updated_active_startups['object_id']:

    if i == last_id:
        id += 1
        #print("\n",True, "last_object_id: ", last_id ," object_id: ", i)
    else:
        id = 0
        #print("\n",False, "last_object_id: ", last_id ," object_id: ", i)


        print("\n index comes here: ", active_startups.at[
            active_startups[active_startups['object_id'] == i].index.values.astype(int)[id], 'is_first_round'])

    last_id = i


# get the max number of funding rounds, so we need to add an extra column for each possible funding round
max_fund_rounds = updated_active_startups['funding_rounds'].max()

utils.print_column_head(updated_active_startups)


# print(active_startups.loc[active_startups['object_id'] == i]) access all "i" duplicates


#active_dups = active_startups[active_startups.duplicated(subset='object_id', keep=False)]
'''







'''
ANOTHER FILE
preprocessing.py code ABOVE

'''

'''

find the same count of duplicates as in closed_startups


max_operating_dups = int(round(operating_startups_weight*closed_startups_dups))
max_acquired_dups = int(round(acquired_startups_weight*closed_startups_dups))
max_ipo_dups = int(round(ipo_startups_weight*closed_startups_dups))

print("\nop weight: ", operating_startups_weight, "\nop amount: ",  round(operating_startups_weight*max_num_active_startups),"\n", "op dups: ", operating_startups.iloc[0:1345].duplicated(subset=dataset_key).sum())


while(operating_startups_dups > closed_startups_dups):
    active_startups_max_rows -= 1
    operating_startups_dups = operating_startups.iloc[0:active_startups_max_rows].duplicated(subset=dataset_key).sum()

operating_startups = operating_startups.iloc[0:active_startups_max_rows]

# the problem is that it gets to the number of duplicates, but then the number of rows is less than the amount we wanted
dups = operating_startups.duplicated(subset=dataset_key).sum()

while (dups > closed_startups_dups):
    max_num_active_startups -= 1
    dups = operating_startups.iloc[0:max_num_active_startups].duplicated(subset=dataset_key).sum()

operating_startups = operating_startups.iloc[0:max_num_active_startups]
'''