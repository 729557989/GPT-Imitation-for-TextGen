import torch
from word_sequence import Word2Sequence

# data path
path = 'input.txt'


# Model Configuration
max_len = 256 # Max number of words per input array
d_model = 384 # How many dimensions of characteristics used to describe each word embedding
batch_size = 32 # Batch Size
vocab_size = 7666 # Total num of recorded vocabs in tokenizer
nheads = 8 # Number of Attention Heads for Transformer
dim_feedforward = 128 # Size of neurons for each Decoder Block final feedforward
decoder_layers = 6 # Number of Decoder Block


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
GPT = "GPT.pt"

# Load tokenizer
tokenizer = Word2Sequence()
tokenizer.load_dict(w2s_dict_path)