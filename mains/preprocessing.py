import pandas as pd
import numpy as np
from utils import utils as pp

startups = pd.read_csv("../datasets/CrunchBase_MegaDataset/startups.csv")
rounds = pd.read_csv("../datasets/CrunchBase_MegaDataset/funding_rounds.csv")

cleaned_startups = startups.drop(columns=['entity_type','entity_id', 'parent_id', 'name',
                                              'permalink', 'domain',
                                              'homepage_url', 'twitter_username','logo_url',
                                              'logo_width', 'logo_height','short_description',
                                              'relationships','created_by','created_at','updated_at',
                                          'region','first_investment_at','last_investment_at',
                                          'investment_rounds', 'invested_companies',
                                          'first_milestone_at', 'last_milestone_at'])
# drop startups with no funding
cleaned_startups.dropna(subset=['first_funding_at'], inplace=True)
# format dates
cleaned_startups.founded_at = pd.to_datetime(cleaned_startups.founded_at)
cleaned_startups.first_funding_at = pd.to_datetime(cleaned_startups.first_funding_at)
cleaned_startups.last_funding_at = pd.to_datetime(cleaned_startups.last_funding_at)
# format values such as funds raised
cleaned_startups.funding_total_usd = pd.to_numeric(cleaned_startups.funding_total_usd)
cleaned_startups.funding_rounds = pd.to_numeric(cleaned_startups.funding_rounds)
cleaned_startups.milestones = pd.to_numeric(cleaned_startups.milestones)

'''
 ! We'll have to take into account 2008 financial crisis
 So we remove startups founded before 2003
'''
cleaned_startups.query("founded_at >= 2003", inplace=True)
# subsitute empty cells by NaNs
cleaned_startups.replace({'': np.nan})
# change key name
pp.rename_id_label(cleaned_startups,'id','object_id')

'''
 Add column with boolean status, 
 0 = operating, ipo, acquired, etc, the startup is active
 1 = closed
'''
cleaned_startups.insert(cleaned_startups.columns.get_loc('status'),'status_bool',0)
cleaned_startups.status_bool = np.where(cleaned_startups.status == 'closed', 1,0)

# get funding rounds
cleaned_funding_rounds = rounds.drop(columns=['source_url', 'created_by',
       'created_at', 'updated_at'])
cleaned_funding_rounds.replace({'': 0})
cleaned_funding_rounds.funded_at = cleaned_funding_rounds.funded_at.astype('datetime64[ns]')
cleaned_funding_rounds.raised_amount_usd = pd.to_numeric(cleaned_funding_rounds.raised_amount_usd)
cleaned_funding_rounds.participants = pd.to_numeric(cleaned_funding_rounds.participants)

# get avg time between funding rounds and avg participants
cleaned_funding_rounds.sort_values(by=['object_id','funded_at'], ascending=True,inplace=True)
cleaned_funding_rounds['time_bw_rounds'] = cleaned_funding_rounds.groupby('object_id')['funded_at'].diff().dt.days.fillna(0).astype(int) / 30
averages = pd.DataFrame(columns=['object_id'])
averages['object_id'] = cleaned_funding_rounds.drop_duplicates(subset=['object_id'])['object_id']
averages['avg_time_bw_rounds'] = cleaned_funding_rounds.groupby('object_id')['time_bw_rounds'].mean().tolist()
averages['avg_funds_raised_usd'] = cleaned_funding_rounds.groupby('object_id')['raised_amount_usd'].mean().tolist()
averages['avg_participants'] = cleaned_funding_rounds.groupby('object_id')['participants'].mean().tolist()
averages.reset_index(inplace=True)
averages.drop(columns=['index'])

cleaned_df = cleaned_startups.merge(averages,on='object_id')
cleaned_df.to_csv("../datasets/cleaned_startups.csv")
