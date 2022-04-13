1. Use GPT to Generate Text
2. Data: Shakespeare

In Pytorch,
    the src_mask is the triangular tril_mask [sequence_length, sequence_length]
    src_padding_mask is just pad_mask [sequence_length, sequence_length]

3 components of GPT embedding:

    1) word embedding: [batch_size, sent_length, d_model]
        consisting of word vectors

    2) segment embedding: [batch_size, max_len, d_model]
        consisting of embeddings of ->
            0 (padding elements)
            1 (any element that're part of sentence 1)
            2 (any element that're part of sentence 2)
            3 (any element that're part of sentence 3)
            ... an so on

    3) positional embedding (parameter weights): [max_len, d_model]
        trainable randomly initialized embedding of size -> [max_len, d_model]


PyTorch by default inputs: [max_len, batch_size, d_model]

# consider: '\n' + '.' OR '\n' + '?' OR '\n' + '!' as a sentence
# Difference between the primitive transformer's decoder and GPT's decoder:
    primitive transformer's decoder has 2 attentions:
        1. multi-head attention (with tril mask)
        2. encoder-deocder attention (with padding mask)
    GPT's decoder as ONLY 1 attention:
        1. multi-head attention (with tril mask)


Max seg is 12



review:
    1. loss funcs
    2. optimizers
    3. activation functions