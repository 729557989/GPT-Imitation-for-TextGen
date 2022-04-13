from nltk.tokenize import WordPunctTokenizer
from word_sequence import Word2Sequence
import config
from torch.utils.data import Dataset, DataLoader
import torch


def get_preprocessed_data(text):
    
    # consider: sequence between each "\n" in input.txt as sentence
    seq_dict = {}
    temp_sequence = []
    count = 0
    for i in range(len(text)):
        if text[i] != "\n":
            temp_sequence += [text[i]]
        else:
            seq_dict[count] = temp_sequence
            count = count + 1
            temp_sequence = []
            
    l = []
    for idx, (key, value) in enumerate(seq_dict.items()):
        tknzed = WordPunctTokenizer().tokenize("".join(value).lower())
        for word in tknzed:
            l.append(word)
    
    inp = []
    for i in range(0, len(l)-config.max_len):
        inp.append(l[i:i+config.max_len])
    
    return inp

class Dataset(Dataset):
  def __init__(self, sequences, max_len, limit=None):
    self.max_len = max_len
    
    # self.sequences = sequences if limit == None else sequences[:limit]
    self.sequences = sequences
    
    self.tokenizer = Word2Sequence()
    self.tokenizer.load_dict(config.w2s_dict_path)
  
  def __getitem__(self, idx):
    x = ["<SOS>"] + self.sequences[idx][:-1]
    y = self.sequences[idx][0:-1] + ["<EOS>"]
    
    x = self.tokenizer.transform(x, max_len=self.max_len, pad_first=False)
    y = self.tokenizer.transform(y, max_len=self.max_len, pad_first=False)

    return x, y

  def __len__(self):
    return len(self.sequences)


def collate_fn(batch):
    '''
    param batch: ([x, y]， [x, y], 一个getitem的结果...)
    '''
    x, y = list(zip(*batch))
    return torch.LongTensor(x), torch.LongTensor(y)

def get_dataloader(dataset, batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            collate_fn=collate_fn)
    return dataloader



text = open(config.path).readlines()
seqs = get_preprocessed_data(text)
dataset = Dataset(seqs, config.max_len)
dataloader = get_dataloader(dataset, config.batch_size)


# if __name__ == "__main__":
#     text = open(config.path).readlines()
#     seqs, segs = get_preprocessed_data(text)
#     print(len(segs[0]))
    
#     dataset = Dataset(seqs, segs, config.max_len)
#     for i, (x, segment, y) in enumerate(dataset):
#         print(x)
#         print(len(x))
#         print(segment)
#         print(len(segment))
#         print(y)
#         print(len(y))
#         break
    
#     dataloader = get_dataloader(dataset, config.batch_size)
#     for i, (x, segment, y) in enumerate(dataloader):
#         print(x)
#         print(x.shape)
#         print(segment)
#         print(segment.shape)
#         print(y)
#         print(y.shape)
#         break