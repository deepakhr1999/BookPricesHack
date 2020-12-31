import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class BookDataset(Dataset):
    def __init__(self, frame:pd.DataFrame):
        super().__init__()
        self.frame = frame.reset_index()
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return {
            col : self.frame[col][idx]
            for col in self.frame
        }
    

def preprocess(train, isTrain=True, transforms=True):
    # extract review value
    train.Reviews = train.Reviews.apply(lambda x: float(x[:3]) )
    
    # first word is num ratings
    train.Ratings = train.Ratings.apply(
        lambda x: float(
                    x.split()[0].replace(',','')
                )
    )

    if transforms:
        train.Ratings = train.Ratings.apply(lambda x: np.log(x))
        
    train.Ratings = train.Ratings.astype('float32')
    train.Reviews = train.Reviews.astype('float32')

    # for Edition just check if Hardcover is there
    train.Edition = train.Edition.apply(lambda x: 1 if 'Hardcover' in x else 0)

    features = 'Author Genre BookCategory Reviews Ratings Edition'.split()
    if isTrain:
        train.Price = train.Price.astype('float32')
        features.append('Price')

    return train[features]

def categoricalToIndices(train, val, test):
    catFeatures = 'Author Genre BookCategory'.split()
    for col in catFeatures:
        # use the intersection for resolving indices
        trainTypes = set(train[col]) 
        testTypes  = set(test[col])
        types = sorted(list(trainTypes.intersection(testTypes)))
        
        wordIdx = {}
        for i, t in enumerate(types):
            wordIdx[t] = i+1

        for frame in (train, val, test):
            if frame is not None:
                frame[col] = frame[col].apply(lambda x: wordIdx[x] if x in wordIdx else 0)
    return

def getBookDataset(trainFile, testFile):
    train = pd.read_excel(trainFile)
    test = pd.read_excel(testFile)

    train = preprocess(train, isTrain=True)
    test  = preprocess(test, isTrain=False)

    train, val = train_test_split(train, test_size=0.2, random_state=69)
    categoricalToIndices(train, val, test)

    
    return (BookDataset(data) for data in (train, val, test))