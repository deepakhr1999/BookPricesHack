# run this as $python -m albert.BookDataset
import math
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BookDataset(Dataset):
    def __init__(self, frame:pd.DataFrame):
        super().__init__()
        self.data = dict(frame.reset_index())
        
    def __len__(self):
        return len(self.data['Title'])

    def __getitem__(self, idx):
        return {
            col : self.data[col][idx]
            for col in self.data
        }

class NameSpace:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=4)
    
def getBookDataset(trainFile, testFile, returnValidation=True):
    train = preprocess(trainFile)
    test  = preprocess(testFile)
    
    if returnValidation:
        train, val = train_test_split(train, test_size=0.2, random_state=69)
    else:
        val = None

    featureSizes = categoricalToIndices(train, test, val)

    return train, val, test, NameSpace(**featureSizes)

def mapAuthors(authors):
    x = authors.split(', ') # authors separated by commas
    x = x[:5] if len(x) > 5 else x # max length is 5
    x = [a for a in x if not a.isnumeric()] # exclude numeric names
    return x

"""Transforms for normalizing columns"""
ratingsTransform = lambda x: (math.log(1+x)/10) ** .5
reviewsTransform = lambda x: (x/5)**4
priceTransform  = lambda x: math.log(1 + x)

def preprocess(filename:str):
    """
        Cleans numeric features and applies transforms
    """
    train = pd.read_excel(filename)
    
    # extract review value
    train.Reviews = train.Reviews.apply(lambda x: x[:3]).astype('float32').apply(reviewsTransform).astype('float32')

    # first word is number of people who left ratings
    train.Ratings = train.Ratings.apply(lambda x: x.split()[0].replace(',','')).astype('float32').apply(ratingsTransform).astype('float32')
    
    # for Edition just check if Hardcover is there
    # a binary feature can be treated as a float instead of embedding
    train.Edition = train.Edition.apply(lambda x: 1 if 'Hardcover' in x else 0).astype('float32')
    
    # separate authors
    authors = list(train.Author.apply(mapAuthors))
    
    # drop the original author column
    train.drop('Author', axis=1, inplace=True)

    # Convert series of lists into a dataframe of columns
    expanded_authors = pd.DataFrame(authors, columns=[f'author_{i}' for i in range(5)])

    # assign new columns to the dataframe 
    train = train.assign(**expanded_authors)

    # replace the Nones in author_i columns by '0'
    train.replace({None: '0'}, inplace=True)

    if 'Price' in train:
        train.Price = train.Price.apply(priceTransform).astype('float32')
    # return the cleaned dataframe
    return train


"""
    Convert to categorical
"""
def getFeatureMap(train, test, featuresToCat):
    """
        Common labels are indexed to non-zero idxs
        Labels in the symmetric difference are mapped to 0
    """
    # gather columns in featuresToCat and stack them into one column
    concatFeatures = lambda frame: set(itertools.chain(*[frame[col] for col in featuresToCat]))
    train, test = concatFeatures(train), concatFeatures(test)

    # common features are encoded to labels, minLabel=1
    common = sorted(list(train.intersection(test)))
    featureMap = {label:idx+1 for idx, label in enumerate(common)}
    
    return featureMap

def categoricalToIndices(train, test, val=None):
    """Encodes categorical variables into indices
        Takes in train, test and val dataframes
    """
    featureSizes = {}
    for col in 'Author Genre BookCategory'.split():
        # Dataframe has 5 author columns
        featuresToCat = [f'author_{i}' for i in range(5)] if col=='Author' else [col]
        
        featureMap = getFeatureMap(train, test, featuresToCat)
        featureSizes[col] = max(featureMap.values()) + 1
        
        for frame in (train, test, val):
            if frame is not None:
                for f in featuresToCat:
                    frame[f] = frame[f].apply(lambda x: featureMap.get(x, 0))
    
    return featureSizes


if __name__ == '__main__':
    print("Preprocessing...", end=' ')
    train, val, test = getBookDataset('Data_Train.xlsx', 'Data_Test.xlsx')
    print("got train, val and test frames of shape: ", train.shape, val.shape, test.shape)

    print("Using albert v2...")
    train = BookDataset(val)
    trainLoader = DataLoader(train, batch_size=69)
    batch = next(iter(trainLoader))
    print(f"Dataset with length {len(train)}")
    print("Batch shapes")
    for key in batch:
        print(key.ljust(10), batch[key].shape)