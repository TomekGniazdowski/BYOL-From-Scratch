from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_byol(
    model: nn.Module,
    train_dl: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
    device: str,
):
    
    model.to(device)
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        for step, (x, _) in enumerate(train_dl):
            x = x.to(device)

            loss_byol = model(x)

            optimizer.zero_grad()
            loss_byol.backward()
            optimizer.step()
            model.update_target_network()
        
            pbar.set_postfix({
                'Epoch': epoch, 
                'Step': step,
                'loss': round(loss_byol.item(), 3)
                })

    return model
    
    