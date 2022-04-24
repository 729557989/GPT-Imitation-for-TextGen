"""
    Description: Data Preprocessing
    Author: Jimmy Lu
    Date: March 2022
"""
from nltk.tokenize import WordPunctTokenizer
from word_sequence import Word2Sequence
import config
from torch.utils.data import Dataset, DataLoader
import torch


def txt_to_wordlist(path):
    """
    Args:
        path (str): File storing text data
    Purpose:
        Preprocess words in the text file to list of words
    Returns:
        A list of all words in the file from path
    """
    text = open(path).readlines()
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
    
    word_list = []
    for idx, (key, value) in enumerate(seq_dict.items()):
        tknzed = [WordPunctTokenizer().tokenize(x) + ["\n"] for x in value]
        for sent in tknzed:
            for word in sent:
                word_list.append(word)
                
    return word_list

def to_sequence(wordlist, max_len):
    """
    Args:
        wordlist (list): List of words of the text file
        max_len (int): Maximum token limit for each returning word sequence
    Purpose:
        Preprocess word list to word sequences of length max_len
    Returns:
        input_sequences (2D list): A list of word list containing words from wordlist of limit max_len
    """
    input_sequences = []
    
    for i in range(0, len(wordlist)-max_len):
        input_sequences.append(wordlist[i:i+max_len])
        
    return input_sequences

class Dataset(Dataset):
    """
    Structure of Data:
        INPUT: ["<SOS>"] + word1 + word2 + word3 + word4 + word5 + word6 (Assume Max_Len was 7)
        TARGET: word1 + word2 + word3 + word4 + word5 + word6 + ["<EOS>"] (Assume Max_Len was 7)
    Purpose:
        PyTorch Dataset mechanism for future training
    """
    def __init__(self, sequences, tokenizer, max_len, limit=None):
        self.max_len = max_len
        
        self.sequences = sequences if limit == None else sequences[:limit]
        
        self.tokenizer = tokenizer
  
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
    Purpose:
        Convert word sequences from PyTorch Dataset to torch.LongTensor
    Param:
        batch: ([x, y]， [x, y], output of getitem...)
    '''
    x, y = list(zip(*batch))
    return torch.LongTensor(x), torch.LongTensor(y)

def get_dataloader(dataset, batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn):
    """
    Args:
        dataset (torch.utils.data.Dataset): PyTorch Dataset containing ready word sequences
        batch_size (int): batch_size of dataset
        shuffle (bool, optional): Whether to shuffle returning dataloader. Defaults to True.
        drop_last (bool, optional): Whether to drop last batch of returning dataloader. Defaults to False.
        collate_fn (_type_, optional): Collate_fn for processing dataset. Defaults to collate_fn.
    Purpose:
        Return final processed PyTorch dataloader for training
    Returns:
        dataloader (torch.utils.data.DataLoader): PyTorch Dataloader containing sentence data
    """
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            collate_fn=collate_fn)
    return dataloader



word_list = txt_to_wordlist(config.path)
input_sequences = to_sequence(word_list, config.max_len)
dataset = Dataset(input_sequences, config.tokenizer, config.max_len, limit=None) # try with full dataset see what happens
dataloader = get_dataloader(dataset, config.batch_size)


# #NOTE: Test Run
# if __name__ == "__main__":
#     word_list = txt_to_wordlist(config.path)
#     input_sequences = to_sequence(word_list, config.max_len)
#     print(len(input_sequences[0]))
    
#     dataset = Dataset(input_sequences, config.max_len, limit=None)
#     for i, (x, y) in enumerate(dataset):
#         print(x)
#         print(len(x))
#         print(y)
#         print(len(y))
#         break
    
#     dataloader = get_dataloader(dataset, config.batch_size)
#     for i, (x, y) in enumerate(dataloader):
#         print(x)
#         print(x.shape)
#         print(y)
#         print(y.shape)
#         break