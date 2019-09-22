

from math import sqrt 
from numpy import dot
from numpy.linalg import norm 
from datetime import datetime


def similarity(prefs, id1, id2):
    si = {}
    for item in prefs[id1]: 
        if item in prefs[id2]:
            si[item] = 1
    # if they have no ratings in common, return 0
    if len(si) == 0: return 0
    # calculate the cosine similarity 
    sim1 = [prefs[id1][item] for item in si]
    sim2 = [prefs[id2][item] for item in si]
    
    if  sum(sim1) == 0 and prefs[id1] == prefs[id2]:
        similarity = 1
        
    elif sum(sim1) == 0 or sum(sim2) == 0:
        similarity = 0
    
    else:
        similarity= dot(sim1, sim2)/(norm(sim1) * norm(sim2))
        
    return similarity


def similarity_Edist(prefs, id1, id2):
    si = {}
    for item in prefs[id1]: 
        if item in prefs[id2]:
            si[item] = 1
    # if they have no ratings in common, return 0
    if len(si) == 0: return 0
        
    sum_of_squares = sum([pow(prefs[id1][item] - prefs[id2][item], 2) for item in prefs[id1] if item in prefs[id2]] )
    similarity = 1/(1+sum_of_squares)
        
    return similarity



def topMatches(prefs, id, n = 100, similarity = similarity):
    scores = [(similarity(prefs, id, other), other) for other in prefs if other != id]
    #sort the list so the highest scores appear at the top 
    scores.sort()
    scores.reverse()
    return scores[0:n]

    

def get_item_similarity(c_prefs, n = 25, similarity = similarity):
    # Create a dictionary of items showing which other items they are most similar to.
    result = {}
    c = 0
    for item in c_prefs:
    # Status updates for large datasets
        c += 1
        # Find the most similar items to this one
        scores=topMatches(c_prefs,item,n=n,similarity=similarity)
        result[item]=scores
    return result 
    

def get_recommendedItems(prefs, itemMatch, id, n_rec= 3, threshold = .01):
    userScores = prefs[id]
    scores = {}
    totalSim = {}
    
    # Loop over items rated by this user
    for (item,score) in userScores.items(  ):
    # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:

      # Ignore if this user has already rated this item
            if item2 in userScores: continue

      # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*score

      # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
 
   # Divide each total score by total weighting to get an average
    rankings=[(score/totalSim[item], item) for item,score in scores.items() if totalSim[item] > 0 and score/totalSim[item] > threshold ]

   # Return the rankings from highest to lowest
    rankings.sort(  )
    rankings.reverse(  )
    if len(rankings) > n_rec:
        return rankings[0:n_rec]
    else: 
        return rankings

            
############################################################################################################
# import similarity, similarity_Edist, topMatches, get_item_similarity, get_recommendedItems

df_util_wide = df_util.pivot(index = 'id', columns = 'item', values ='score')
item2id_dict = {col:df_util_wide[col].dropna().to_dict() for col in df_util_wide}


aa = df_util_wide.to_dict('index')
id2item_dict = {m: {k:v for k,v in aa[m].items() if pd.notnull(v)} for m in aa }
del aa, df_util_wide 

n_rec = len(item2id_dict)
ids = list(id2item_dict.keys())

##calculate the similarity of two id 
similarity(id2item_dict, ids[0], ids[1])
similarity_Edist(id2item_dict, ids[0], ids[1])

#find the top matched ids of a id  
# takes a while to run 
topMatches(id2item_dict ,id = ids[0], n = 100, similarity = similarity)

# get the item similarities 
itemsim=get_item_similarity(item2id_dict, n = n_rec )

# get a recomended items for a specific id, return top 3 score and itmes  
get_recommendedItems(prefs=id2item_dict, itemMatch = itemsim, id = ids[0])

df_util[df_util['id'] == ids[0]]

#final recommendations for each ids for 5 items 
id_recmd = {}
for id in ids:
    id_recmd[id] = get_recommendedItems(prefs=id2item_dict, itemMatch = itemsim, id = id, n_rec= 5)
  
