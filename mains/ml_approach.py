import pandas as pd
import numpy as np
import mains.preprocessing as pp
import sklearn
from sklearn import linear_model,preprocessing
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import GetOldTweets3 as getTw

startups = pd.read_csv("../datasets/cleaned_startups.csv")
# drop columns that we do not want
startups.drop(columns=['Unnamed: 0','overview','city','index','state_code'],inplace=True)
# remove strings frmom object_id
startups['object_id'].replace('c:','', regex=True,inplace=True)

# convert qualitative variables to quantitative ones
pp.convert_to_dummies(startups,'category_code')
pp.convert_to_dummies(startups,'tag_list')

startups.replace({np.nan : 0},inplace=True)
cols = ['funding_total_usd','avg_time_bw_rounds','avg_funds_raised_usd','avg_participants','eco_freedom_index']
startups[cols] = startups[cols].round().astype(int)


# ANALYSIS AND PREDICTION

predict = 'status_bool'
X = startups.drop(['object_id','normalized_name','status','status_bool',
    'founded_at','closed_at','description','country_code','first_funding_at','last_funding_at','milestones','avg_participants'],1)
y = startups[predict]
print("columns: ",X.columns,"\n")

# use of stratify to make sure the proportion of success / fail startups is equal between training and test set
x_train, x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1,stratify=y)

# scale the data
scaler = preprocessing.StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train.values),columns=X.columns)
x_test = pd.DataFrame(scaler.transform(x_test.values),columns=X.columns)

# LOG REGRESSION
logreg = linear_model.LogisticRegression(C=10,solver='lbfgs')
logreg.fit(x_train,y_train)
acc = logreg.score(x_test,y_test)
y_pred = logreg.predict(x_test)
y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = sklearn.metrics.roc_curve(y_test,  y_pred_proba)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
#plt.show()
print("accuracy: ",sklearn.metrics.accuracy_score(y_test,y_pred))
print("precision: ",sklearn.metrics.precision_score(y_test,y_pred))
print("recall: ",sklearn.metrics.recall_score(y_test,y_pred))
print(logreg.predict(x_test))

# KNEAREST NEIGHBORS
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(x_train,y_train)
acc = dt.score(x_test,y_test)
importance = pd.DataFrame({
    'Variable': X.columns,
    'Importance': dt.feature_importances_
}).sort_values(by='Importance',ascending=False)
print(importance.head(20))

text_query = 'wetpaint'
username = 'getvetter'
twCriteria = getTw.manager.TweetCriteria().setUsername(username).setMaxTweets(4)
tweets = getTw.manager.TweetManager.getTweets(twCriteria)
txt_tweets = [[tweet.text] for tweet in tweets]
print("tweets: ",txt_tweets)

