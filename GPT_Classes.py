from MultiHeadAttention import MultiHeadAttention
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_samples = []
        self.output_samples = []

        token_ids = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_samples.append(torch.tensor(token_ids[i:i+max_length]))
            self.output_samples.append(torch.tensor(token_ids[i+1:i+max_length+1]))
    
    def __len__(self):
        return len(self.input_samples)

    def __getitem__(self, idx):
        return self.input_samples[idx], self.output_samples[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

torch.manual_seed(123)
class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        pass

    def forward(self, x : torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x-mean)/(torch.sqrt(variance + self.epsilon))
        return x*self.scale + self.shift
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.op_pipeline = nn.Sequential(nn.Linear(input_dim,4*input_dim), GELU(), nn.Linear(4*input_dim, input_dim))
    
    def forward(self, x):
        return self.op_pipeline(x)
    
class TransformerBlock(nn.Module):

    def __init__(self, hyper_param):
        super().__init__()
        self.norm1 = LayerNorm(hyper_param["emb_dim"])
        self.norm2 = LayerNorm(hyper_param["emb_dim"])
        self.attention_block = MultiHeadAttention(hyper_param["emb_dim"], hyper_param["emb_dim"], hyper_param["context_length"], hyper_param["drop_rate"], hyper_param["n_heads"], hyper_param["qkv_bias"])
        self.Dropper = nn.Dropout(hyper_param["drop_rate"])
        self.ffn = FeedForward(hyper_param["emb_dim"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention_block(x)
        x = self.Dropper(x)
        x = x+shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.Dropper(x)
        x = x+shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.token_emb = nn.Embedding(hyper_params["vocab_size"], hyper_params["emb_dim"])
        self.positional_emb = nn.Embedding(hyper_params["context_length"], hyper_params["emb_dim"])
        self.dropper = nn.Dropout(hyper_params["drop_rate"])
        
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(hyper_params) for i in range(hyper_params["n_layers"])])
        
        self.final_norm = LayerNorm(hyper_params["emb_dim"])
        self.out_layer = nn.Linear(hyper_params["emb_dim"], hyper_params["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_emb(in_idx)
        pos_embeds = self.positional_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.dropper(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_layer(x)
        return logits

def text_generater(model, text_data : torch.Tensor, context_size, max_new_tokens):
    for i in range(max_new_tokens):
        model_input = text_data[:, -context_size:]
        with torch.no_grad():
            logits = model(model_input)
        
        generative_logits = logits[:,-1,:]
        generative_logits = torch.softmax(generative_logits, dim = -1)
        generated_tokens = torch.argmax(generative_logits, dim=-1, keepdim=True)
        text_data = torch.cat((text_data, generated_tokens), dim = 1)
    
    return text_data