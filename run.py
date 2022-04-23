from GPT import GPT, generate, train
import torch
import config



if __name__ == "__main__":
    # NOTE: Configurate GPT model
    gpt = GPT(
        vocab_size = config.vocab_size,
        max_len = config.max_len,
        d_model = config.d_model,
        nhead = config.nheads,
        dim_feedforward = config.dim_feedforward,
        num_layers = config.decoder_layers
    )
    gpt.to(config.device)
    # NOTE: Load Pretrained GPT Model
    gpt.load_state_dict(torch.load(config.GPT, map_location=config.device))
    
    # NOTE: Run Text Generation by providing header
    header = "First Citizen:"
    print(type(gpt))
    generated = generate(gpt, config.tokenizer, header, k=1, temp=0.7)
    print(" ".join(config.tokenizer.inverse_transform(generated.cpu()[0], is_tensor=True)))