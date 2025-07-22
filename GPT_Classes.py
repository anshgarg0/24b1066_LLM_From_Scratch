from MultiHeadAttention import MultiHeadAttention
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import numpy as np

def text_to_tokens(text, tokeniser: tiktoken.Encoding):
    tokens = tokeniser.encode(text, allowed_special={"|<endoftext>|"})
    return torch.tensor(tokens).unsqueeze(0)

def tokens_to_text(tokens : torch.Tensor, tokeniser: tiktoken.Encoding):
    return tokeniser.decode(tokens.squeeze(0).tolist())

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

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError("Shapes no equal")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.positional_emb.weight = assign(gpt.positional_emb.weight, params['wpe'])
    gpt.token_emb.weight = assign(gpt.token_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attention_block.W_query.weight = assign(
            gpt.transformer_blocks[b].attention_block.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].attention_block.W_key.weight = assign(
            gpt.transformer_blocks[b].attention_block.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].attention_block.W_value.weight = assign(
            gpt.transformer_blocks[b].attention_block.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].attention_block.W_query.bias = assign(
            gpt.transformer_blocks[b].attention_block.W_query.bias, q_b)
        gpt.transformer_blocks[b].attention_block.W_key.bias = assign(
            gpt.transformer_blocks[b].attention_block.W_key.bias, k_b)
        gpt.transformer_blocks[b].attention_block.W_value.bias = assign(
            gpt.transformer_blocks[b].attention_block.W_value.bias, v_b)

        gpt.transformer_blocks[b].attention_block.out_proj.weight = assign(
            gpt.transformer_blocks[b].attention_block.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].attention_block.out_proj.bias = assign(
            gpt.transformer_blocks[b].attention_block.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].ffn.op_pipeline[0].weight = assign(
            gpt.transformer_blocks[b].ffn.op_pipeline[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].ffn.op_pipeline[0].bias = assign(
            gpt.transformer_blocks[b].ffn.op_pipeline[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].ffn.op_pipeline[2].weight = assign(
            gpt.transformer_blocks[b].ffn.op_pipeline[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].ffn.op_pipeline[2].bias = assign(
            gpt.transformer_blocks[b].ffn.op_pipeline[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_layer.weight = assign(gpt.out_layer.weight, params["wte"])

device = torch.device("cuda")

def calc_loss_1(inputs : torch.Tensor, expected_outputs, model):
    inputs, expected_outputs = inputs.to(device), expected_outputs.to(device)
    logits = model(inputs)
    return torch.nn.functional.cross_entropy(logits.flatten(0,1), expected_outputs.flatten(0))

def calc_loss_loader(dataloader, model, num_batches = None):
    total_loss = 0
    if len(dataloader)==0:
        return total_loss
    elif num_batches==None:
        num_batches = len(dataloader)
    else:
        num_batches = min(len(dataloader), num_batches)
    for i, (inputs, targets) in enumerate(dataloader):
        if i<num_batches:
            total_loss+=calc_loss_1(inputs, targets, model).item()
        else:
            break

    return total_loss/num_batches

def calc_loss(input_msg, targets, model, device):
    input_msg, targets = input_msg.to(device), targets.to(device)
    logits = model(input_msg)
    loss = torch.nn.functional.cross_entropy(logits[:,-1,:], targets)
    return loss

def calculate_accuracy(loader, model, device, num_batches=None):
    model.to(device)
    model.eval()
    if len(loader) == 0:
        return
    elif num_batches == None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    total_messages_seen, correct_classifications = 0, 0
    for j, (inputs, targets) in enumerate(loader):
        if j<num_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            total_messages_seen += inputs.shape[0]
            with torch.no_grad():
                correct_classifications += (torch.argmax(model(inputs)[:, -1, :], dim = -1) == targets).sum().item()
        else:
            break
    model.train()
    return correct_classifications/total_messages_seen

def train_loop(train_loader, val_loader, model, device, num_epochs, optimiser : torch.optim.AdamW, val_freq, eval_iter):
    model.train()
    model.to(device)
    num_batches = len(train_loader)
    
    batches_seen = -1
    messages_seen_yet = 0
    train_losses, val_losses, train_acc, val_acc, messages_seen = [], [], [], [], []
    for i in range(num_epochs):
        for j, (inputs, targets) in enumerate(train_loader):
            if j < num_batches:
                optimiser.zero_grad()
                loss = calc_loss(inputs, targets, model, device)
                loss.backward()
                optimiser.step()

                batches_seen += 1
                messages_seen_yet += len(inputs)
            else:
                break

            if batches_seen % val_freq == 0:
                with torch.no_grad():
                    train_loss = evaluate_model(model, train_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_loss = evaluate_model(model, val_loader, device, eval_iter)
                    val_losses.append(val_loss)
                    messages_seen.append(messages_seen_yet)
                print(f"Epoch {i}, Batch {batches_seen}, train_loss {train_loss}, val_loss {val_loss}")
        train_acc.append(calculate_accuracy(train_loader, model, device, eval_iter))
        val_acc.append(calculate_accuracy(val_loader, model, device, eval_iter))
        print(f"Training accuracy: {train_acc[-1]*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_acc[-1]*100:.2f}%")

    return train_losses, val_losses, messages_seen, train_acc, val_acc

def evaluate_model(model, loader, device, num_batches):
    model.eval()
    if len(loader) == 0:
        return
    elif num_batches == None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    total_loss = 0
    for j, (inputs, targets) in enumerate(loader):
        if j<num_batches:
            total_loss += calc_loss(inputs, targets, model, device)
        else:
            break
    model.train()
    return total_loss/num_batches



def training_function(model, dataloader_training, validating_dataset, num_epochs, optimiser : torch.optim.AdamW, device, eval_iter, sample, tokeniser):
    model.to(device)
    tokens_seen = 0
    val_freq = 5
    global_step = 0
    train_losses, val_losses, track_tokens_seen = [],[],[]
    for i in range(num_epochs):
        model.train()
        for inputs, targets in dataloader_training:
            optimiser.zero_grad() #clear out previous gradients
            loss = calc_loss_1(inputs, targets, model)
            loss.backward()
            optimiser.step()
            tokens_seen += inputs.numel()
            global_step += 1

            if global_step%val_freq==0:
                train_loss, valid_loss = evaluate_model_1(model, dataloader_training, validating_dataset, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(valid_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {i} Step: {global_step} Training loss: {train_loss} Validation Loss: {valid_loss} ")
        model.eval()
        print(tokens_to_text(text_generater(model, text_to_tokens(sample, tokeniser).to(device), 256, 50), tokeniser).replace('\n', " "))
    return train_losses, val_losses, track_tokens_seen

def evaluate_model_1(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, eval_iter)
    model.train()
    return train_loss, val_loss

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()
