import pandas as pd
import numpy as np
import itertools
    
def extend_authors(train:pd.DataFrame):
    """Max 5 authors per book"""
    authors = list(train.Author.apply(lambda x: x.split(', ')))

    # # there is one test example with 7 authors. 
    for i in range(len(authors)):
        if len(authors[i]) > 5: # truncate till 5
            authors[i] = authors[i][:5]
        authors[i] = [x for x in authors[i] if not x.isnumeric()]
    
    authorColumns = [f'author_{i}' for i in range(5)]
    expanded_authors = pd.DataFrame(authors, columns=authorColumns)
    train = train.assign(**expanded_authors)
    train = train.drop('Author', axis=1)

    for f in authorColumns:
        train[f] = train[f].apply(lambda x: '0000' if x is None else x)

    return train

def preprocess(train, isTrain=True, transforms=True):
    # extract review value
    train.Reviews = train.Reviews.apply(lambda x: float(x[:3]) )
    
    # first word is num ratings
    train.Ratings = train.Ratings.apply(
        lambda x: float(
                    x.split()[0].replace(',','')
                )
    )

    # for Edition just check if Hardcover is there
    train.Edition = train.Edition.apply(lambda x: 1 if 'Hardcover' in x else 0)

    features = 'Genre BookCategory Reviews Ratings Edition'.split()
    train = extend_authors(train)
    features += [f'author_{i}' for i in range(5)]

    if isTrain:
        features.append('Price')

    return train[features]


def authorToIndices(train, test):
    features = [f'author_{i}' for i in range(5)]
    
    trainAutors = set(itertools.chain(*[
        train[f] for f in features
    ]))

    testAutors = set(itertools.chain(*[
        test[f] for f in features
    ]))

    types = sorted(list(trainAutors.intersection(testAutors)))
    
    wordIdx = {t: i+1 for i, t in enumerate(types)}
    
    for frame in (train, test):
        for f in features:
            frame[f] = frame[f].apply(lambda x: wordIdx.get(x, 0))

    return

def categoricalToIndices(train, test):
    
    authorToIndices(train, test)

    catFeatures = 'Genre BookCategory'.split()
    for col in catFeatures:
        # use the intersection for resolving indices
        trainTypes = set(train[col]) 
        testTypes  = set(test[col])
        types = sorted(list(trainTypes.intersection(testTypes)))
        
        wordIdx = {t: i+1 for i, t in enumerate(types)}
        
        for frame in (train, test):
            frame[col] = frame[col].apply(lambda x: wordIdx.get(x, 0))

    return

def getBookDataset(trainFile, testFile, transforms=True):
    train = pd.read_excel(trainFile)
    test = pd.read_excel(testFile)

    train = preprocess(train, isTrain=True, transforms=transforms)
    test  = preprocess(test, isTrain=False, transforms=transforms)

    categoricalToIndices(train, test)

    
    return train, test