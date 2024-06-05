import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def accuracy_metric(outputs: torch.Tensor, labels: torch.Tensor):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

def train(
    model: nn.Module, 
    optimizer: torch.optim,
    criterion: nn.Module,
    train_dl: DataLoader,
    num_epochs: int,
    device: str = 'cuda'
    ):
    
    model.to(device)
    model.train()
    pbar = tqdm(range(num_epochs))
    
    for epoch in pbar:
        epoch_outputs, epoch_labels = [], []
        epoch_loss = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_outputs += outputs.tolist()
            epoch_labels += y.tolist()
            epoch_loss += loss.item()
       
        acc = accuracy_metric(torch.Tensor(epoch_outputs), torch.Tensor(epoch_labels))
        pbar.set_postfix({
            'Epoch': epoch, 
            'loss': round(epoch_loss, 3),
            'accuracy': round(acc, 3)
        })
    
    return model

def test(
    model: nn.Module, 
    test_dl: DataLoader,
    device: str = 'cuda',
    print_acc: bool = True
):
    
    model.to(device)
    model.eval()
    all_outputs, all_labels = [], []
    
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            all_outputs += outputs.tolist()
            all_labels += y.tolist()
        acc = accuracy_metric(torch.Tensor(all_outputs), torch.Tensor(all_labels))
        if print_acc:
            print(f'Accuracy of the network on the test images: {acc}')
        return acc