import torch
import numpy

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

    