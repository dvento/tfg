import pandas as pd
import numpy as np
import tensorflow as tf
import functools

'''

DATA FORMAT
- Dates: YEAR-MONTH-DAY

'''


# Define the unique key for all dataset entries
dataset_key = 'object_id'


# Rename labels for a selected dataframe aka columns
def rename_id_label(dataframe, old_label,new_label):
    dataframe.rename(columns={old_label: new_label}, inplace=True)


# Convert key column to string and remove first to characters (as they are a letter followed by ":")
def standardize_heading(dataframe):
    dataframe[dataset_key] = dataframe[dataset_key].astype(str).str[2:]


def update_columns(dataframe):
    '''

    New dataframe where we have one object_id per row

    '''

    # updated_active_startups = pd.DataFrame(columns=active_startups.columns)
    dataframe.drop(columns=['pre_money_valuation_usd', 'pre_money_valuation', 'pre_money_currency_code',
                            'post_money_valuation_usd', 'post_money_valuation', 'post_money_currency_code'],
                   inplace=True)

    # new columns for each fund round
    fund_rounds_cols = list(
        ['funding_round_id', 'funded_at', 'funding_round_type', 'funding_round_code', 'raised_amount_usd',
         'raised_amount',
         'raised_currency_code', 'is_first_round', 'is_last_round', 'participants', 'source_description'])
    fund_rounds_cols = fund_rounds_cols * 9

    old_col_index = list(dataframe.columns)
    new_col_index = old_col_index[1:22] + old_col_index[29:36] + old_col_index[37:39] + old_col_index[
                                                                                        36:37] + old_col_index[
                                                                                                 39:40] + fund_rounds_cols + old_col_index[
                                                                                                                             43:49] + old_col_index[
                                                                                                                                      22:29]
    return new_col_index

    # updated_active_startups.to_csv('../datasets/updated_active_startups.csv',encoding='utf-8')


def main():

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

    # rename id column to match rest of csv files
    rename_id_label(startups,'id',dataset_key)
    rename_id_label(acquistions,'acquiring_object_id',dataset_key)
    rename_id_label(investments, 'funded_object_id',dataset_key)

    # convert from object to string and remove the chars of the key
    standardize_heading(startups)
    standardize_heading(funding_rounds)
    standardize_heading(degrees)


    # testing how to drop columns
    funding_rounds = funding_rounds.drop(columns=['source_url'])

    startups_cleaned = startups.drop(columns=['entity_type','entity_id', 'parent_id', 'name',
                                              'permalink', 'domain',
                                              'homepage_url', 'twitter_username','logo_url',
                                              'logo_width', 'logo_height','short_description',
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
    max_num_active_startups = 2135#len(closed_startups.index) GOT THIS NUMBER BY GUESSTIMATION

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
    print("active startups lenght: ", len(active_startups.index), "\n Active duplicated: ", active_startups.duplicated(subset=dataset_key).value_counts())

    pass

if __name__ == "__main__":
    main()
