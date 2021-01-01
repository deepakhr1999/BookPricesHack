import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
ALBERT_END_IDX = 3
ALBERT_MAX_LEN = 512

def mapAuthors(authors):
    x = authors.split(', ') # authors separated by commas
    x = x[:5] if len(x) > 5 else x # max length is 5
    x = [a for a in x if not a.isnumeric()] # exclude numeric names
    return x

def preprocess(train:pd.DataFrame, isTrain=True):
    """
        Only cleans numeric features
        Transforms are to be applied later
    """
    # extract review value
    train.Reviews = train.Reviews.apply(lambda x: x[:3]).astype('float32')

    # first word is number of people who left ratings
    train.Ratings = train.Ratings.apply(lambda x: x.split()[0].replace(',','')).astype('float32')
    
    # for Edition just check if Hardcover is there
    # a binary feature can be treated as a float instead of embedding
    train.Edition = train.Edition.apply(lambda x: 1 if 'Hardcover' in x else 0).astype('float32')
    
    # separate authors
    authors = train.Author.apply(mapAuthors)
    
    # drop the original author column
    train.drop('Author', axis=1, inplace=True)

    # Convert series of lists into a dataframe of columns
    expanded_authors = pd.DataFrame(authors, columns=[f'author_{i}' for i in range(5)])

    # assign new columns to the dataframe 
    train = train.assign(**expanded_authors)

    # replace the Nones in author_i columns by '0'
    train.replace({None: '0'}, inplace=True)

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
    for col in 'Author Genre BookCategory'.split():
        # Dataframe has 5 author columns
        featuresToCat = [f'author_{i}' for i in range(5)] if col=='Author' else [col]
        featureMap = getFeatureMap(train, test, featuresToCat)
        for frame in (train, test, val):
            if frame is not None:
                for f in featuresToCat:
                    frame[f] = frame[f].apply(lambda x: featureMap.get(x, 0))

    return

"""Convert a series of sentences into embeddings"""
def albertTokenize(tokenizer, model, batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True)
    max_length = inputs['input_ids'].size(1)
    if max_length > ALBERT_MAX_LEN:
        for key in inputs:
            inputs[key] = inputs[key][:, :ALBERT_MAX_LEN]
        # for those that are truncated, input_ids[:, 511] != 0
        last_ids = inputs['input_ids'][:, ALBERT_MAX_LEN - 1]
        # we set them to index corresponding to the end token
        inputs['input_ids'][last_ids != 0, ALBERT_MAX_LEN - 1] = ALBERT_END_IDX
    
    inputs = {k: inputs[k].to(model.device) for k in inputs}
    return inputs

def albertForward(tokenizer, model, sentences, desc=None)->torch.Tensor:
    loader = DataLoader(sentences, batch_size=50)
    with torch.no_grad():
        outputs = []
        for batch in tqdm(loader, desc=desc):
            inputs = albertTokenize(tokenizer, model, batch)
            for key in inputs:
                print(key, inputs[key].shape)
            out = model(**inputs).pooler_output.cpu()
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        return outputs
