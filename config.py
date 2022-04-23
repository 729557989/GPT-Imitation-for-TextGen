import torch
from word_sequence import Word2Sequence

# data path
path = 'input.txt'

# Model Configuration -> GPT.pt (Best Params my laptop can endure without running out of memory)
max_len = 150
d_model = 768
batch_size = 8
vocab_size = 13360
nheads = 12
dim_feedforward = 384
decoder_layers = 12
lr = 1e-4

epoch = 1

# tokens for preprocessing
PAD_TAG = "<PAD>"
SOS_TAG = "<SOS>"
EOS_TAG = "<EOS>"
UNK_TAG = "<UNK>"
PAD = 0
SOS = 1
EOS = 2
UNK = 3

# Dictionary Json path for tokenizer
w2s_dict_path = "w2s_dict/dict.json"

# Configure GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path of pretrained/saved GPT model
GPT = "Pretrained/GPT.pt"

# Load tokenizer
tokenizer = Word2Sequence()
tokenizer.load_dict(w2s_dict_path, notify=True)