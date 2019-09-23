
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
    for chan in c_prefs:
    # Status updates for large datasets
        c += 1
        # Find the most similar items to this one
        scores=topMatches(c_prefs,chan,n=n,similarity=similarity)
        result[chan]=scores
    return result 
    

def get_recommendation(prefs, id, similarity = similarity, n_rec = 3, threshold = 0.01):
    totals = {}
    simSums = {}
    for other in prefs:
        # don't compare me to myself
        if other==id: continue
        sim=similarity(prefs,id,other)
        
        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:    
          # only score movies I haven't seen yet
          if item not in prefs[id] or prefs[id][item]==0:
            # Similarity * Score
            totals.setdefault(item,0)
            totals[item]+=prefs[other][item]*sim
            # Sum of similarities
            simSums.setdefault(item,0)
            simSums[item]+=sim    
    
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items() and total/simSums[item] > threshold]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings[0:n_rec]

%time get_recommendation(prefs= prefs, id = 123, similarity = similarity)

