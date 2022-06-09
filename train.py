"""
    Description: Sample Training
    Author: Jimmy Lu
    Date: March 2022
"""
from data import dataloader
import torch
from GPT import GPT, generate, train
import config



if __name__ == "__main__":
    # # NOTE: You may run this to check the data structure
    # for i, (x,y) in enumerate(dataloader):
    #     print(x[0])
    #     print(y[0])
    #     break
    # # print(" ".join(config.tokenizer.inverse_transform(x[1], is_tensor=True)))
    # # print(" ".join(config.tokenizer.inverse_transform(y[1], is_tensor=True)))
    
    # NOTE: Initialize model and parameters here
    gpt = GPT(
        vocab_size = config.vocab_size,
        max_len = config.max_len,
        d_model = config.d_model,
        nhead = config.nheads,
        dim_feedforward = config.dim_feedforward,
        num_layers = config.decoder_layers
    )
    gpt.to(config.device)
    # # NOTE: If Transfer Learning, make sure the HyperParameters match
    # gpt.load_state_dict(torch.load(config.GPT, map_location=config.device))
    
    
    # NOTE: TRAINING!!!
    train(
        gpt, dataloader, config.batch_size,
        config.device, saving_path=config.GPT,
        epoch=config.epoch, lr=config.lr
    )
    
    
    # NOTE: Test model by generating!
    header = "First Citizen:"
    generated = generate(gpt, config.tokenizer, header, k=3, temp=0.7)
    print(" ".join(config.tokenizer.inverse_transform(generated.cpu()[0], is_tensor=True)))