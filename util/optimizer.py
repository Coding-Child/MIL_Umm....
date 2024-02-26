import torch.optim as optim
from util.scheduler import NoamLR

def select_optimizer(model, lr: float = 1e-3, opt: str = 'Adam'):
    optimizer=None

    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    elif opt == 'SGD':    
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-9)
    elif opt == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-9)
    else:
        raise ValueError(f"Unsupported optimizer: {opt}")
    
    return optimizer

def select_scheduler(optimizer, train_loader, gamma, d_model, warmup_steps, scheduler: str = 'Noam'):
    if scheduler == 'Noam':
        scheduler = NoamLR(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    elif scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=gamma)
    elif scheduler is None:
        scheduler = None
    
    return scheduler