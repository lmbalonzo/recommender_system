import pandas as pd
import numpy as np
import operator

col_names=['User ID', 'Movie ID', 'Rating']
col_names2=['Movie ID', 'Title']
file=pd.read_csv('./archive/rating.csv', sep=',', names=col_names, usecols=range(3), skiprows=1, nrows=11100)
file2=pd.read_csv('./archive/movie.csv', sep=',', names=col_names2, usecols=range(2), skiprows=1)
merged=pd.merge(file2,file)

# Group by Movie ID, Compute Total Number of Ratings and Average Rating per Movie
movIdGroup = file.groupby('Movie ID').agg({'Rating': [np.size,np.mean]})
# Normalize Number of Ratings / How Popular It is Compared to Other Movies
movNumRat = pd.DataFrame(movIdGroup['Rating']['size'])
norMovNumRat=movNumRat.apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x)))

# Put Everything in Dictionary {'Movie ID': 'Title', 'Popularity', 'Mean Rating'}
movDict={}
for row in range(file2.shape[0]):
    movieID=file2.iloc[row]['Movie ID']
    title=file2.iloc[row]['Title']
    try:
        popularity=norMovNumRat.loc[movieID].get('size')
    except KeyError:
        popularity=0
    try:
        meanRating=movIdGroup.loc[movieID]['Rating']['mean']   
    except KeyError:
        meanRating=0
    movDict[movieID] = (title, popularity, meanRating)

# Calculate Difference of 2 Movies Based on Popularity
def calc_dist(x,y):
    pop1 = x[2]
    pop2 = y[2]
    popDist = abs(pop1-pop2)
    return popDist

#Testing
K=10
avg=0
test_IDset=list(set(file['User ID'].values.tolist()))
total = 0
hits = 0

for test_ID in test_IDset:
    #Movies watched:
    watched=pd.DataFrame(merged.loc[merged['User ID'] == test_ID])
    #Movies unwatched:
    unwatched=pd.concat([watched['Movie ID'],file2['Movie ID']]).drop_duplicates(keep=False)
    #Merged and Sorted Dataframe with Columns: Movie ID, Title, User ID, Rating, Popularity
    merged2=pd.merge(watched,norMovNumRat,right_index=True, left_on='Movie ID').sort_values(['Rating', 'size'],ascending=[False, False] )
    #Test Movie is the Top Rated and Most Popular Movie by Test User
    testMov=merged2.iloc[0]['Movie ID']

    allmov=list(set(merged['Movie ID'].values.tolist()))
    allmov=[x for x in allmov if x != testMov]
 
    dist=[]
    #for mov in unwatched:
    for mov in allmov:
        d = calc_dist(movDict[testMov], movDict[mov])
        dist.append((mov,d))
    dist.sort(key=operator.itemgetter(1))
    neigh = []
    for i in range(K):
        avg += movDict[dist[i][0]][2]
        neigh.append([dist[i][0], movDict[dist[i][0]][0], dist[i][1], movDict[dist[i][0]][2]])
    avg /= float(K)
    neigh_df=pd.DataFrame(neigh,columns=['Movie ID', 'Title', 'Distance', 'Average Rating'])
    print("Recommendations: \n",neigh_df)
    print("User: ",test_ID)
    print("Top Watched Movie: ", movDict[testMov][0])
    print("Predicted Rating (Average of Neighbors): ", avg)
    print("Actual Rating: ", movDict[testMov][2])

    mer_reco_wat=pd.merge(neigh_df, watched)
    print(mer_reco_wat)
    hits+=mer_reco_wat.shape[0]
    total+=K
    print("Hits: ", hits)
    print("Total: ", total)
    print("Hit Rate: ", hits/total)
