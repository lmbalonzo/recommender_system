from logging.handlers import WatchedFileHandler
from pickle import TRUE
import pandas as pd
import numpy as np
import operator
from pandas_datareader import test

col_names=['User ID', 'Movie ID', 'Rating']
col_names2=['Movie ID', 'Title']
file=pd.read_csv('./archive/rating.csv', sep=',', names=col_names, usecols=range(3), skiprows=1, nrows=11100)
file2=pd.read_csv('./archive/movie.csv', sep=',', names=col_names2, usecols=range(2), skiprows=1)
merged=pd.merge(file2,file)

# Group by Movie ID, Compute Total Number of Ratings and Average Rating per Movie
movIdGroup = merged.groupby('Title').agg({'Rating': [np.size,np.mean]})
# Normalize Number of Ratings / How Popular It is Compared to Other Movies
movNumRat = pd.DataFrame(movIdGroup['Rating']['size'])
norMovNumRat=movNumRat.apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x)))

# Make Pivot Table (User on Rows, Movies on Columns)
norMat=merged.pivot_table(index=['User ID'], columns=['Title'], values='Rating')
corrMat=norMat.corr(method='pearson', min_periods=5)
for column in corrMat.columns:
    corrMat[column] = (corrMat[column] - corrMat[column].min()) / (corrMat[column].max() - corrMat[column].min())

# Store Similarity Scores Per Movie in Dictionary
allmov=merged['Title'].values.tolist()
sim_score={}
for mov in allmov:
    sim_score[mov]=corrMat[[mov]].reset_index().rename(columns={mov:'Similarity'}).dropna() #Compare Only With Movies with Correlation Values

#Testing
K=10
nSimilar=5
avg=0
test_IDset=list(set(file['User ID'].values.tolist()))
total = 0
hits = 0

for test_ID in test_IDset:
    print("User ID: ", test_ID)
    #Movies watched:
    watched=pd.DataFrame(norMat.loc[test_ID].dropna(axis=0, how='all')\
        .sort_values(ascending=False))\
        .reset_index()\
        .rename(columns={test_ID:'Rating'})
    pred_dict={}
    for mov in allmov:
        try:
            watched_sim = pd.merge(left=watched, right=sim_score[mov], on='Title', how='inner').sort_values('Similarity', ascending=False)[:nSimilar]
            pred_rat=round(np.average(watched_sim['Rating'], weights=watched_sim['Similarity']), 6)
        except ZeroDivisionError:
            pred_rat=0
        poplar=norMovNumRat.loc[mov].values[0]
        pred_dict[mov]=(pred_rat,poplar)
        sorted_pred=sorted(pred_dict.items(), key=lambda k: (k[1][0],k[1][1]), reverse=True)[:K]
    neigh = []
    for i in range(K):
        df_MovieID=file2.loc[file2['Title']==sorted_pred[i][0]]['Movie ID'].values[0]
        df_Title=sorted_pred[i][0]
        df_Pred=sorted_pred[i][1][0]
        neigh.append([df_MovieID, df_Title, df_Pred])
    avg  /=float(K)  
    neigh_df=pd.DataFrame(neigh,columns=['Movie ID', 'Title', 'Predicted Rating'])
    print("Recommendations: \n", neigh_df)
    mer_reco_wat=pd.merge(neigh_df, watched)
    print(mer_reco_wat)
    hits+=mer_reco_wat.shape[0]
    total+=K
    print("Hits: ", hits)
    print("Total: ", total)
    print("Hit Rate: ", hits/total)