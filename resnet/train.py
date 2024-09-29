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


def get_data(num_train: int) -> list[np.ndarray | torch.Tensor]:
    data = get_CIFAR10_data(num_train, 50000 - num_train, 10000, divide_std=False)
    for k, v in data.items():
        if 'test' in k:
            continue
        # CIFAR-10 is small, load the entire train and val dataset onto device
        if k.startswith('X'):
            data[k] = torch.as_tensor(v, dtype=data_type, device=device)
        else:
            data[k] = torch.as_tensor(v, dtype=torch.int64, device=device)
    return data.values()


@torch.no_grad()
def check_accuracy(model, X, y, num_samples=1000, batch_size=100):
    """
    Adapted from assignment2.Solver.check_accuracy
    """
    # Maybe subsample the data
    N = X.shape[0]
    if num_samples < N:
        mask = torch.randint(N, size=(num_samples,), device=device)
        X = X[mask]
        y = y[mask]

    # Compute predictions in batches
    assert num_samples % batch_size == 0
    y_pred = []
    model.eval()
    for start in range(0, num_samples, batch_size):
        scores = model(X[start: start+batch_size])
        y_pred.append(torch.max(scores, dim=1).indices)
    y_pred = torch.cat(y_pred)
    acc = torch.mean(y_pred == y, dtype=torch.float)
    return acc


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
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(num_train)

    optimizer = optim.SGD(model.parameters(), config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['decay_steps'], gamma=0.1)
    model.train()

    best_val_acc = 0.1
    best_model_dict = None
    ckpt_name = None

    for t in range(config['total_steps']):
        indices = torch.randint(0, num_train, size=(batch_size,), device=device)
        X, y = X_train[indices], y_train[indices]

        scores = model(X)
        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # step optimizer
        scheduler.step(t)
        current_lr = scheduler.get_last_lr()[0]

        first_it = t == 0
        last_it = t == num_train - 1
        if first_it or last_it or t % args.eval_interval == 0:
            model.eval()
            train_acc = check_accuracy(model, X_train, y_train)
            val_acc = check_accuracy(model, X_val, y_val)
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
        model.eval()

        best_val_acc = check_accuracy(best_model, X_val, y_val, num_samples=50000-num_train, batch_size=250)
        print(f"Best validation accuracy is {best_val_acc:.4f}")

        X_test = torch.as_tensor(X_test, device=device, dtype=data_type)
        y_test = torch.as_tensor(y_test, device=device, dtype=torch.int64)
        best_test_acc = check_accuracy(best_model, X_test, y_test, num_samples=len(X_test), batch_size=250)
        print(f"Best test accuracy is {best_test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--eval_interval", "-ei", type=int, default=1000)
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
