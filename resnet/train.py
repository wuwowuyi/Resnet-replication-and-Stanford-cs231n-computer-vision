import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler
from torchvision.transforms import v2 as T
import torchvision.datasets as dset

from assignment2.cs231n.data_utils import get_CIFAR10_data
from resnet.resnet_model import ResNet

USE_GPU = True
data_type = torch.float32
ckpt_path = Path(__file__).parent / 'checkpoint'

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ResNet(3)  # 20 layers
model.to(device=device, dtype=data_type)


def get_dataloader(num_train, batch_size):
    transforms_train = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(size=(32, 32), padding=4),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar10_train = dset.CIFAR10('./data', train=True, download=True, transform=transforms_train)
    loader_train = DataLoader(cifar10_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(num_train)))
    cifar10_val = dset.CIFAR10('./data', train=True, download=True, transform=transform_val)
    loader_val = DataLoader(cifar10_val, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))
    cifar10_test = dset.CIFAR10('./data', train=False, download=True, transform=transform_val)
    loader_test = DataLoader(cifar10_test, batch_size=batch_size)
    return loader_train, loader_val, loader_test


@torch.no_grad()
def check_accuracy(resnet, data_loader, num_batches=None):
    """
    Adapted from assignment2.Solver.check_accuracy
    """
    resnet.eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device=device, dtype=data_type), y.to(device=device, dtype=torch.int64)
        scores = resnet(x)
        y_pred = torch.max(scores, dim=1).indices
        correct += torch.sum(y == y_pred)
        total += y.shape[0]
        if num_batches is not None and i >= num_batches:
            break
    return correct / total


def save_checkpoint(step: int, values: dict, ckpt_name=None) -> None:
    if ckpt_name is None:
        ckpt_name = f"resnet_{str(int(time.time()))}"
    filename = f"{ckpt_name}_{step}.pkl"
    with open(ckpt_path / filename, "wb") as f:
        pickle.dump(values, f)


def train(config: dict, args: argparse.Namespace):

    seed = 1337 + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_train, batch_size = config['num_train'], config['batch_size']
    loader_train, loader_val, loader_test = get_dataloader(num_train, batch_size)

    optimizer = optim.SGD(model.parameters(), config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['decay_steps'], gamma=0.1)
    model.train()

    best_val_acc = 0.1
    best_model_dict = None
    ckpt_name = None

    num_epoch = config['total_steps'] // (num_train // batch_size)
    t = 0

    for epoch in range(num_epoch):
        for x, y in loader_train:
            x, y = x.to(device=device, dtype=data_type), y.to(device=device, dtype=torch.int64)

            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step optimizer
            scheduler.step(t)
            t += 1
            current_lr = scheduler.get_last_lr()[0]

            first_it = t == 1
            if first_it or t % args.eval_interval == 0:
                y_pred = torch.max(scores, dim=1).indices
                train_acc = torch.sum(y == y_pred) / y.shape[0]
                model.eval()
                val_acc = check_accuracy(model, loader_val, 10)
                model.train()

                if args.wandb_log:
                    wandb.log({
                        "step": t,
                        "loss": loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "lr": current_lr,
                    })

                # save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_dict = model.state_dict()
                    ckpt_name = 'resnet_best_model_ckpt'

                if ckpt_name is not None or t % args.checkpoint_interval == 0:
                    values = {
                        'step': t,
                        'learning_rate': current_lr,
                        'model_dict': model.state_dict()
                    }
                    save_checkpoint(t, values, ckpt_name)
                    ckpt_name = None

    if best_model_dict:
        best_model = ResNet(3)
        best_model.load_state_dict(best_model_dict)
        best_model.to(device=device, dtype=data_type)
        best_model.eval()

        best_val_acc = check_accuracy(best_model, loader_val)
        print(f"Best validation accuracy is {best_val_acc:.4f}")

        best_test_acc = check_accuracy(best_model, loader_test)
        print(f"Best test accuracy is {best_test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--eval_interval", "-ei", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=16000)  # total_steps // 4
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_log", action="store_true")
    args = parser.parse_args()

    assert args.checkpoint_interval >= args.eval_interval

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.wandb_log:  # wandb logging
        wandb_project = 'restnet-cifar10'
        wandb_run_name = str(int(time.time()))
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    print("training parameters:")
    for k, v in dict(**config, **vars(args)).items():
        print(f"{k}: {v}")

    train(config, args)


if __name__ == "__main__":
    main()
