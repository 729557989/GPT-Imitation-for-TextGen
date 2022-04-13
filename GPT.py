import config
from nltk.tokenize import WordPunctTokenizer
import torch
import torch.nn as nn
import random



def get_key_padding_mask(data, pad_token=config.PAD):
    attentio_mask = data==pad_token
    return attentio_mask


def select_top_k(predictions, current_loc, k=1):
    predicted_index = random.choice(
        predictions[0, current_loc, :].sort(descending=True)[1][:k]).item()
    return predicted_index


def generate(model, tokenizer, x, k=1):
    target = ['<SOS>'] + WordPunctTokenizer().tokenize(x.lower())
    pred_loc = len(target) - 1
    target = tokenizer.transform(target, max_len=128, pad_first=False)
    target = torch.LongTensor(target).unsqueeze(0)
    
    for i in range(config.sent_length-pred_loc):
        target = target.to(config.device)
        out = model(target)
        pred = select_top_k(out, pred_loc, k=k)
        if pred == 2:
            i -= 1
            continue
        target[0][loc] = pred
        loc += 1
    return target


class GPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        
        # [7666 (vocab_size), 384 (d_model)]
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        # [256 (max_len), 384 (d_model)]
        self.pos_embed = nn.Parameter(torch.randn(max_len, d_model, device=config.device) / 10)
        
        # attention (tril) mask
        self.attention_mask = torch.triu(torch.ones(
                (max_len, max_len),
                dtype=torch.long,
                device=config.device
            ),diagonal=1)
        
        self.attention_mask = self.attention_mask == 1
        
        # So that the mask isn't part of backprop, it should be constants
        self.register_buffer("mask", self.attention_mask)
        
        # GPT Decoder Block
        self.DecoderBlock = Decoder(d_model, nhead, dim_feedforward, num_layers)
        
        # output feed forward network
        self.FINAL_ffn = nn.Linear(in_features = (d_model), out_features = vocab_size)
        
    def forward(self, x):
        # [32 (batch_size), 256 (max_len)]
        pad_mask_x = get_key_padding_mask(x)

        
        # all shape: [32 (batch_size), 256 (max_len), 384 (d_model)]
        word_embedding = self.embed(x)
        
        # shape: [32 (batch_size), 256 (max_len), 384 (d_model)]
        embedded_x = word_embedding + self.pos_embed
        
        
        # shape: [32 (batch_size), 256 (max_len), 384 (d_model)]
        output = self.DecoderBlock(embedded_x, self.attention_mask, pad_mask_x)
        
        output = self.FINAL_ffn(output) # [32 (batch_size), 256 (max_len), 7666 (vocab_size)]
        
        return output
        
class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.2):
        super().__init__()
        
        # GPT Decoder Layer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = 'relu'
        )
        norm = nn.LayerNorm(normalized_shape = d_model)

        # 定义编码器
        self.Decoder = nn.TransformerEncoder(encoder_layer = decoder_layer,
                                             num_layers = num_layers,
                                             norm = norm)
        
    def forward(self, x, attention_mask_x, pad_mask_x):
        # Convert to PyTorch input formats / 转换成torch要求的格式
        # [32 (batch_size), 256 (max_len), 384 (d_model)] -> [256 (max_len), 32 (batch_size), 384 (d_model)]
        x = x.permute(1, 0, 2)
        
        out = self.Decoder(src=x, mask=attention_mask_x, src_key_padding_mask=pad_mask_x)
    
        # [256 (max_len), 32 (batch_size), 384 (d_model)] -> [32 (batch_size), 256 (max_len), 384 (d_model)]
        out = out.permute(1, 0, 2)
        
        return out