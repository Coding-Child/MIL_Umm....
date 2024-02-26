import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import torch.optim as opt

from sklearn.model_selection import KFold
from torchvision import transforms as T

from model.ResNetMIL import ResNetMIL
from Dataset.MILDataset import PathologyDataset, collate_fn, data_split
from trainer.ModelTrainer import train, seed_everything
from trainer.ModelEvaluator import test_model
from util.gaussian_blur import GaussianBlur
from util.optimizer import select_optimizer, select_scheduler

import matplotlib.pyplot as plt
import os
import matplotlib
import pandas as pd
matplotlib.use('Agg')


def main(args):
    seed_everything(42)

    lr = args.learning_rate
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    num_fc = args.num_fc
    num_classes = args.num_classes
    num_patience = args.num_patience
    pretrained = args.pretrained
    dropout = args.dropout
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_sample = args.num_instance
    csv_root = args.csv_root_dir
    loss_weight = args.loss_weight
    sch = args.scheduler
    optim = args.optimizer
    gamma = args.gamma
    warmup_steps = args.warmup_steps
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    transform_train = T.Compose([T.RandomHorizontalFlip(p=0.5),
                                 T.RandomApply([GaussianBlur(kernel_size=int(0.1 * 224))], p=0.2),
                                 T.RandomApply([T.ColorJitter(brightness=(1 - 0.16, 1 + 0.16), 
                                                              contrast=(1 - 0.11, 1 + 0.11), 
                                                              saturation=(1 - 0.16, 1 + 0.16))], p=0.5),
                                 T.ToTensor(),
                                 T.Normalize(mean=[0.753, 0.475, 0.696], std=[0.195, 0.224, 0.173])
                                 ])
    transform_test = T.Compose([T.ToTensor(),
                                T.Normalize(mean=[0.753, 0.475, 0.696], std=[0.195, 0.224, 0.173])
                                ])

    data = pd.read_csv(csv_root)

    criterion_bag = nn.BCELoss()
    criterion_instance = nn.CrossEntropyLoss(label_smoothing=0.1)

    for i, (trn_val_idx, ts_idx) in enumerate(kf.split(data)):
        print('-' * 35)
        print(f'Fold {i + 1} Start')
        print('-' * 35)

        model = ResNetMIL(d_model=d_model, 
                          num_heads=num_heads, 
                          num_layers=num_layers, 
                          dropout=dropout, 
                          num_fc=num_fc, 
                          pretrained=pretrained, 
                          num_classes=num_classes)
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])
        model.cuda()

        train_val_data = data.iloc[trn_val_idx]
        test_data = data.iloc[ts_idx]

        train_size = int(len(train_val_data) * 0.01)
        val_size = len(train_val_data) - train_size

        train_val_dataset = PathologyDataset(df=train_val_data, num_samples=num_sample, transform=transform_train)
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
        test_dataset = PathologyDataset(df=test_data, num_samples=num_sample, transform=transform_test)

        train_loader = DataLoader(val_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=True, 
                                  drop_last=True, 
                                  collate_fn=collate_fn)
        dev_loader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=4,
                                pin_memory=True, 
                                drop_last=True, 
                                collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=4, 
                                 pin_memory=True, 
                                 drop_last=True, 
                                 collate_fn=collate_fn)

        optimizer = select_optimizer(model=model, lr=lr, opt=optim)
        scheduler = select_scheduler(optimizer=optimizer, train_loader=train_loader, gamma=gamma, d_model=d_model, warmup_steps=warmup_steps, scheduler=sch)

        instance_loss_epoch, bag_loss_epoch, val_instance_loss, val_bag_loss = train(model=model,
                                                                                     train_loader=train_loader,
                                                                                     val_loader=dev_loader,
                                                                                     criterion1=criterion_instance,
                                                                                     criterion2=criterion_bag,
                                                                                     optimizer=optimizer,
                                                                                     scheduler=scheduler,
                                                                                     num_epochs=num_epochs,
                                                                                     num_patience=num_patience,
                                                                                     loss_weight=loss_weight,
                                                                                     fold=i
                                                                                     )

        os.makedirs('LossGraph', exist_ok=True)
        epochs = range(1, len(instance_loss_epoch) + 1)
        plt.plot(epochs, instance_loss_epoch, label='instance pred loss', color='r', marker='o')
        plt.plot(epochs, bag_loss_epoch, label='bag pred loss', color='b', marker='o')
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xlim(1, len(instance_loss_epoch))
        plt.xticks(range(1, len(instance_loss_epoch) + 1))
        plt.savefig(f'LossGraph/Loss_fold_{i + 1}.jpg', facecolor='#eeeeee')
        plt.close()

        plt.plot(epochs, val_instance_loss, label='instance pred loss', color='r', marker='o')
        plt.plot(epochs, val_bag_loss, label='bag pred loss', color='b', marker='o')
        plt.title("Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xlim(1, len(instance_loss_epoch))
        plt.xticks(range(1, len(instance_loss_epoch) + 1))
        plt.savefig(f'LossGraph/Val_Loss_fold_{i + 1}.jpg', facecolor='#eeeeee')
        plt.close()


        test_model(model=model,
                   data_loader=test_loader,
                   criterion_1=criterion_instance,
                   criterion_2=criterion_bag,
                   fold=i
                   )
