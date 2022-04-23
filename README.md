1. Use an imitated GPT model to Generate Text
2. Data: Shakespeare

In Pytorch,
    the src_mask is the triangular tril_mask [sequence_length, sequence_length]
    src_padding_mask is just pad_mask [sequence_length, sequence_length]

2 components of the imitated GPT embedding: (According to the paper, there's also this segment embedding?)

    1) word embedding: [batch_size, sent_length, d_model]
        consisting of word vectors

    3) positional embedding (parameter weights): [max_len, d_model]
        trainable randomly initialized embedding of size -> [max_len, d_model]


PyTorch by default inputs: [max_len, batch_size, d_model]

# Difference between the primitive transformer's decoder and GPT's decoder:
    primitive transformer's decoder has 2 attentions:
        1. multi-head attention (with tril mask + padding mask)
        2. encoder-deocder attention (with padding mask)
    GPT's decoder as ONLY 1 attention:
        1. multi-head attention (with tril mask + padding mask)
        2. GPT restructured the Decoder steps

# Structure of Data:
    INPUT: ["<SOS>"] + word1 + word2 + word3 + word4 + word5 + word6 (Assume Max_Len was 7)
    TARGET: word1 + word2 + word3 + word4 + word5 + word6 + ["<EOS>"] (Assume Max_Len was 7)

# How to Generate Text:
    INPUT: ["<SOS>"] + input headers + ["<PAD>] ...

review:
    1. loss funcs
    2. optimizers
    3. activation functions

# Note: GPTs are strong by stacking HyperParameters and the number of Decoder Blocks,
#       This meant that GPTs are very computationally expensive + picky, so most likely you use pretrained GPTs via HuggingFace