import pandas as pd
import numpy as np
import tensorflow as tf
import functools

'''

DATA FORMAT
- Dates: YEAR-MONTH-DAY

'''

csv_files = ['acquisitions', 'degrees','funding_rows', 'funds','investments', 'ipos','milestones','startups','offices','people']

acquistions = pd.read_csv("../datasets/CrunchBase_MegaDataset/acquisitions.csv")
degrees = pd.read_csv("../datasets/CrunchBase_MegaDataset/degrees.csv")
funding_rounds = pd.read_csv("../datasets/CrunchBase_MegaDataset/funding_rounds.csv")
funds = pd.read_csv("../datasets/CrunchBase_MegaDataset/funds.csv")
investments = pd.read_csv("../datasets/CrunchBase_MegaDataset/investments.csv")
ipos = pd.read_csv("../datasets/CrunchBase_MegaDataset/ipos.csv")
milestones = pd.read_csv("../datasets/CrunchBase_MegaDataset/milestones.csv")
startups = pd.read_csv("../datasets/CrunchBase_MegaDataset/startups.csv")
offices = pd.read_csv("../datasets/CrunchBase_MegaDataset/offices.csv")
people = pd.read_csv("../datasets/CrunchBase_MegaDataset/people.csv")

# Define the unique key for all dataset entries
dataset_key = 'object_id'


# Print labels for a selected dataframe
def print_column_head(dataframe):
    for column in dataframe.columns.values:
        print(column)


# Rename labels for a selected dataframe
def rename_id_label(dataframe, old_label):
    dataframe.rename(columns={old_label: dataset_key}, inplace=True)


# Convert key column to string and remove first to characters (as they are a letter followed by ":")
def standardize_heading(dataframe):
    dataframe[dataset_key] = dataframe[dataset_key].astype(str).str[2:]


# rename id column to match rest of csv files
rename_id_label(startups,'id')
rename_id_label(acquistions,'acquiring_object_id')
rename_id_label(investments, 'funded_object_id')

# convert from object to string and remove the chars of the key
standardize_heading(startups)
standardize_heading(funding_rounds)
standardize_heading(degrees)


# testing how to drop columns
funding_rounds = funding_rounds.drop(columns=['source_url'])

startups_cleaned = startups.drop(columns=['entity_type','entity_id', 'parent_id', 'name',
                                          'permalink','category_code', 'domain',
                                          'homepage_url', 'twitter_username','logo_url',
                                          'logo_width', 'logo_height','short_description',
                                          'state_code','city','region','invested_companies',
                                          'relationships'])

# join all files into one dataframe
joined_data = pd.merge(startups_cleaned,funding_rounds, on=dataset_key)#.merge(degrees,on=dataset_key)

#startups_cleaned.join(funding_rounds,on=dataset_key)

joined_data['founded_at'] = pd.to_datetime(joined_data['founded_at'])


'''
 ! We'll have to take into account 2008 financial crisis
 So we remove startups founded before 2005
'''
joined_data.query("founded_at >= 2005", inplace=True)


# drop startup with no funding info
joined_data.replace({'funding_total_usd' : {None : np.nan, 0 : np.nan}} , inplace=True)
joined_data.dropna(subset=['funding_rounds','funding_total_usd'], inplace=True)


#print_column_head(funding_rounds)
# get only closed startups
closed_startups = joined_data.query("status == 'closed'")

#closed_startups.drop_duplicates(subset=['object_id'], inplace=True)

closed_startups.to_csv('../datasets/closed_startups.csv', encoding='utf-8')

#print(closed_startups[[dataset_key,'funding_rounds','funding_total_usd']])


'''

Here we make the active_startups sample size equal the closed_startups sample size

'''

#active_startups = joined_data.query('status in ["operating","acquired","ipo"]')
operating_startups = joined_data.query('status == "operating"')
acquired_startups = joined_data.query('status == "acquired"')
ipo_startups = joined_data.query('status == "ipo"')

# set the maximum amount of active startups (sum of operating, acquired and ipo), so it matches the number of failed startups aka 'closed'
max_num_active_startups = len(closed_startups.index)

operating_startups_len = len(operating_startups.index)
acquired_startups_len = len(acquired_startups.index)
ipo_startups_len = len(ipo_startups.index)

total_amount_startups = operating_startups_len+acquired_startups_len+ipo_startups_len

operating_startups_weight = (operating_startups_len / total_amount_startups)
acquired_startups_weight = (acquired_startups_len / total_amount_startups)
ipo_startups_weight = (ipo_startups_len / total_amount_startups)

operating_startups = operating_startups.iloc[0:int(round(max_num_active_startups*operating_startups_weight))]
acquired_startups = acquired_startups.iloc[0:int(round(max_num_active_startups*acquired_startups_weight))]
ipo_startups = ipo_startups.iloc[0:int(round(max_num_active_startups*ipo_startups_weight))]

'''

active_startups contains operating, acquired and ipo startups, each category is weighted with respect to the total 
amount of startups and it's correspondent category. It is a file for comparison to closed_startups. 

'''
active_startups = pd.concat([operating_startups,acquired_startups, ipo_startups], ignore_index=True)

# generate csv file with all active startups
active_startups.to_csv('../datasets/active_startups.csv',encoding='utf-8')

# count each number of each state
print("active startups lenght: ", len(active_startups.index))




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


