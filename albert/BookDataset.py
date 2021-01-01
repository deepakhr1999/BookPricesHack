# run this as $python -m albert.BookDataset
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from . import utils

class BookDataset(Dataset):
    def __init__(self, frame:pd.DataFrame):
        super().__init__()
        self.data = dict(frame.reset_index())

        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        model = AutoModel.from_pretrained("albert-base-v2") #11683584 params
        model.eval()
        # model.cuda()

        for textColumn in 'Synopsis Title'.split():
            sentences = list(self.data[textColumn])
            self.data[textColumn] = utils.albertForward(tokenizer, model, sentences, 'Loading '+textColumn)

    def __len__(self):
        return len(self.data['Title'])

    def __getitem__(self, idx):
        return {
            col : self.data[col][idx]
            for col in self.data
        }

def getBookDataset(trainFile, testFile):
    train = pd.read_excel(trainFile)
    test = pd.read_excel(testFile)

    train = utils.preprocess(train, True)
    test  = utils.preprocess(test, False)
    train, val = train_test_split(train, test_size=0.2, random_state=69)

    utils.categoricalToIndices(train, test, val)

    return train, val, test

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