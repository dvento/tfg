import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import utils as pp
import sklearn
from sklearn import linear_model,preprocessing
from sklearn.tree import DecisionTreeClassifier

startups = pd.read_csv("../datasets/CrunchBase_MegaDataset/startups.csv")
startups['description'] = startups['description'].combine_first(startups['short_description'])
#print(startups[startups['short_description'].isnull()]['description'].isnull())
print(startups['twitter_username'].isnull().sum())
startups.dropna(subset=['short_description','category_code','country_code'],inplace=True)
print("\n",startups.shape,"\n")

print(startups.columns)
#print("\n",startups[['twitter_username','domain']])
#print("\n",startups['status'].value_counts())
