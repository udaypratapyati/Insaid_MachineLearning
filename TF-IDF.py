#%%
import pandas as pd

documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'

# bag containing words in the respecitve documents
documentA = documentA.split(' ')
documentB = documentB.split(' ')

# total no. of unique words in both docs
uniqueWords = set(documentA).union(set(documentB))

#%%
# set the count to be zero initially {word : wordcount}
wordCountInDocA = dict.fromkeys(uniqueWords, 0)

# from bag1 get each word and update the word freq
for word in documentA:
    wordCountInDocA[word] += 1

# repeat for bag2
wordCountInDocB = dict.fromkeys(uniqueWords, 0)
for word in documentB:
    wordCountInDocB[word] += 1

#%%
def computeTF(wordCount, bagOfWord):
    '''
    Returns a diction having tf for each word in the record

    tf = freq of word in doc/ total no. of words in doc
    '''
    tfDict = {}
    totalWordCount = len(bagOfWord)

    for word, count in wordCount.items():
        tfDict[word] = count/float(totalWordCount)

    return tfDict
        
tfForDocA = computeTF(wordCountInDocA, documentA)
tfForDocB = computeTF(wordCountInDocB, documentB)

#%%
import math
def computeIDF(list_docWordCount):
    '''
    Returns a diction having idf for each word

    idf = log(Total no. of docs / no. of docs containing the word)
    '''
    N = len(list_docWordCount) # no. of documents

    # create a dic of all key words and init with 0 {word : no. of docs having the word}
    idfDict = dict.fromkeys(list_docWordCount[0].keys(), 0)

    # Iterate through each dict, and update the global dict
    for doc in list_docWordCount:
        for word, count in doc.items():
            if count > 0:
                idfDict[word] += 1

    for word, noOfDocsHavingWord in idfDict.items():
        idfDict[word] = math.log(N/float(noOfDocsHavingWord))

    return idfDict

idfs_forAllDocs = computeIDF([wordCountInDocA, wordCountInDocB])

#%%
def computeTFIDF(tfs, idfs):
    '''
    tfidf = tf * idf
    '''
    tfidfs = {}

    for word, tf in tfs.items():
        tfidfs[word] = tf * idfs[word]

    return tfidfs

tfidfA = computeTFIDF(tfForDocA, idfs_forAllDocs)
tfidfB = computeTFIDF(tfForDocB, idfs_forAllDocs)

#%%
pd.DataFrame([tfidfA, tfidfB])

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'

model = TfidfVectorizer() # initialize model

# this will give row & col which have non zero values only
vectors = model.fit_transform([documentA, documentB])
vectors = vectors.todense() # will insert 0 for row&col which had 0 as values
features = model.get_feature_names() # get the names of the features
vectors = vectors.tolist() # optional

pd.DataFrame(vectors, columns=features)