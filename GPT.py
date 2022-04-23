from nltk.tokenize import WordPunctTokenizer
import torch.nn as nn
import config
import random
import torch
from tqdm import tqdm



class GPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        
        # [7666 (vocab_size), 768 (d_model)]
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        # [150 (max_len), 768 (d_model)]
        self.pos_embed = nn.Parameter(torch.randn(max_len, d_model, device=config.device) / 10)
        
        # attention (tril) mask
        self.attention_mask = torch.triu(
            torch.ones(
                (max_len, max_len),
                dtype = torch.long,
                device = config.device
            )
        , diagonal=1)
        
        self.attention_mask = self.attention_mask == 1
        
        # So that the mask isn't part of backprop, it should be constants
        self.register_buffer("mask", self.attention_mask)
        
        # GPT Decoder Block
        self.DecoderBlock = Decoder(d_model, nhead, dim_feedforward, num_layers)
        
        # output feed forward network
        self.FINAL_ffn = nn.Linear(in_features = d_model, out_features = vocab_size)
        
    def forward(self, x):
        # [8 (batch_size), 150 (max_len)]
        pad_mask_x = get_key_padding_mask(x)

        # all shape: [8 (batch_size), 150 (max_len), 768 (d_model)]
        word_embedding = self.embed(x)
        
        # shape: [8 (batch_size), 150 (max_len), 768 (d_model)]
        embedded_x = word_embedding + self.pos_embed
        
        # shape: [8 (batch_size), 150 (max_len), 768 (d_model)]
        output = self.DecoderBlock(embedded_x, self.attention_mask, pad_mask_x)
        
        output = self.FINAL_ffn(output) # [8 (batch_size), 150 (max_len), 7666 (vocab_size)]
        
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
            activation = 'gelu'
        )
        norm = nn.LayerNorm(normalized_shape = d_model)

        self.Decoder = nn.TransformerEncoder(encoder_layer = decoder_layer,
                                             num_layers = num_layers,
                                             norm = norm)
        
    def forward(self, x, attention_mask_x, pad_mask_x):
        # Convert to PyTorch input formats
        # [8 (batch_size), 150 (max_len), 768 (d_model)] -> [150 (max_len), 8 (batch_size), 768 (d_model)]
        x = x.permute(1, 0, 2)
        
        out = self.Decoder(src=x, mask=attention_mask_x, src_key_padding_mask=pad_mask_x)
    
        # [150 (max_len), 8 (batch_size), 768 (d_model)] -> [8 (batch_size), 150 (max_len), 768 (d_model)]
        out = out.permute(1, 0, 2)
        
        return out
    
    
def get_key_padding_mask(data, pad_token=config.PAD):
    """
    Args:
        data (torch.LongTensor): The Decoder input of shape: [batch_size, max_len]
        pad_token (_type_, optional): The Padding Token Defaults to config.PAD.
    Purpose:
        Get Padding Mask for the transformer
    Returns:
        attentio_mask (2D Torch.Tensor): Padding Mask of shape -> [batch_size, max_len], True for yes pad, False otherwise
    """
    attentio_mask = data==pad_token
    return attentio_mask


def select_top_k(predictions, current_loc, k=1):
    """
    Args:
        predictions (torch.tensor): Model prediction of shape: [1, maxlen, vocab_size]
        current_loc (int): Current word location the model aims to predict
        k (int, optional): How many random options. Defaults to 1.
                           (Say 10, then select randomly from top 10 of highest possibility predictions)
    Purpose:
        Apply Top_K sampling for text generation predictions
    Returns:
        predicted_index (int): location of selected prediction (or word token for output tokenizer)
    """
    temped_pred = predictions[0, current_loc, :]
    predicted_index = random.choice(
        temped_pred.sort(descending=True)[1][:k]
    ).item()
    
    return predicted_index


def generate(model, tokenizer, x, k=1, temp=0.7): # pay attention to loc that's grabbing the word
    """
    Args:
        model (<class 'GPT.GPT'>): The imitated GPT model
        tokenizer (class: word_sequence.Word2Sequence): The tokenizer
        x (str): Text Gen Header Prompt
        k (int, optional): Top k random sampling Defaults to 1.
        temp (float, optional): Control temperature Text Gen technique Defaults to 0.7.
    Purpose:
        Perform Text Generation with the model
    Returns:
        target (torch.tensor): Then generated sequence tokens
    """
    target = ['<SOS>'] + WordPunctTokenizer().tokenize(x.lower())
    pred_loc = len(target)
    target = tokenizer.transform(target, max_len=config.max_len, pad_first=False)
    target = torch.LongTensor(target).unsqueeze(0)
    
    for i in range(config.max_len - pred_loc - 1):
        target = target.to(config.device)
        out = model(target)
        # temperature generation technique
        out = out / temp
        # top k sampling generation technique
        pred = select_top_k(out, pred_loc-1, k=k)
        if pred == 2: # If we encountered <EOS>, we repredict because it causes model to opt out early
            i -= 1
            continue
        target[0][pred_loc] = pred
        pred_loc += 1
    return target


def train(model, dataloader, batch_size, device=None, saving_path=None, epoch=1, lr=1e-4):
    """
    Args:
        model (<class 'GPT.GPT'>): The imitated GPT model
        dataloader (torch.utils.data.Dataset): dataloader containg the dataset
        batch_size (int): batch_size
        device (toch.device, optional): Utilize cuda/GPU during training if available. Defaults to None.
        saving_path (str, optional): Path to save model (end with .pt) Defaults to None.
        epoch (int, optional): Num training epochs Defaults to 1.
        lr (int, optional): Learning rate for the model training Defaults to 1e-4.
    Purpose:
        Perform model training and model saving along the way
    """
    # Will have optimizer and loss func set in stones accordingly to the paper
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epoch):
        model.train()
        total_train_loss = 0
        for _, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader.dataset)/batch_size):
        # for i, (x, y) in enumerate(dataloader):
            x = x.to(device) if device != None else x
            y = y.to(device) if device != None else x
            
            # We essentially consider location of predicted index before padding as prediction
            # Then insert that prediction to the next up coming padding index as prediction and so on
            
            optim.zero_grad()

            # [batch_size, max_len] -> [batch_size, max_len, vocab_size]
            pred_y = model(x)
            
            # loss_func([shape from batch_size * max_len, vocab_size], [shape from batch_size * max_len])
            loss = loss_func(pred_y.reshape(-1, config.vocab_size), y.reshape(-1))

            loss.backward()
            
            optim.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(dataloader)
        
        print(f"EPOCH: {epoch}", f"AVG.Loss: {avg_train_loss}")
        total_train_loss = 0
        
        torch.save(model.state_dict(), saving_path)