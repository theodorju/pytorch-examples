import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from neps.utils.common import load_checkpoint, save_checkpoint
from neps_global_utils import set_seed, process_trajectory
from main import loss_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def load_mnist(batch_size, valid_size, val_test_batch_size=1024, debug=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    train_dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="../data", train=False, transform=transform)

    n_train = len(train_dataset)
    indices = list(range(n_train))
    split = int(np.floor(valid_size * n_train))
    train_idx = indices[:n_train - split]
    valid_idx = indices[n_train-split:]

    # if debug only use a subsample of MNIST
    if debug:
        train_idx = train_idx[:5_000]
        valid_idx = valid_idx[:1_000]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    validation_dataloader = DataLoader(dataset=train_dataset, batch_size=val_test_batch_size, sampler=valid_sampler, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=val_test_batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader

def evaluate_accuracy(model, data_loader, criterion):
    set_seed()
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss += criterion(recon_batch, data, mu, logvar).item()
    loss /= len(data_loader.dataset)
    return loss

def load_mnist(batch_size, valid_size, val_test_batch_size=1024, debug=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="../data", train=False, transform=transform)

    n_train = len(train_dataset)
    indices = list(range(n_train))
    split = int(np.floor(valid_size * n_train))
    train_idx = indices[:n_train - split]
    valid_idx = indices[n_train-split:]

    # if debug only use a subsample of MNIST
    if debug:
        train_idx = train_idx[:5_000]
        valid_idx = valid_idx[:1_000]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    validation_dataloader = DataLoader(dataset=train_dataset, batch_size=val_test_batch_size, sampler=valid_sampler, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=val_test_batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


def adaptive_gradient_clipping(model, clip_factor=10, percentile=95, max_history=100):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not hasattr(param, 'grad_history'):
                param.grad_history = [] 
            param.grad_history.append(param.grad.norm().item())
            param.grad_history = param.grad_history[-max_history:]

            if param.grad_history:
                mean_norm = np.mean(param.grad_history)
                std_norm = np.std(param.grad_history)
                percentile_norm = np.percentile(param.grad_history, percentile)
                
                threshold = min(
                    percentile_norm, 
                    mean_norm + clip_factor * std_norm
                )
                
                current_grad_norm = param.grad.norm()
                if current_grad_norm > threshold:
                    param.grad.mul_(threshold / (current_grad_norm + 1e-6))


def train_epoch(model, optimizer, criterion, train_loader, validation_loader):
    model.train()
    # train for all batches of data in an epoch
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # criterion returns a float
        loss = criterion(recon_batch, data, mu, logvar)
        loss.backward()
        adaptive_gradient_clipping(model, clip_factor=5)
        optimizer.step()
    val_loss = evaluate_accuracy(model, validation_loader, criterion)
    return val_loss


def run_pipeline(
        pipeline_directory,
        previous_pipeline_directory,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        epoch=10 # 10 default if not handled by the searcher
):
    start = time.time()
    # for mf algorithms
    epochs = int(epoch)

    criterion = loss_function
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)

    train_loader, validation_loader, test_loder = load_mnist(
        batch_size=128, valid_size=0.2, debug=False
    )

    previous_state = load_checkpoint(
        directory=previous_pipeline_directory,
        model=model,
        optimizer=optimizer
    )

    start_epoch = previous_state["epochs_trained"] if previous_state is not None else 0
    val_losses = list()

    for ep in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(ep + 1, epochs).ljust(2))
        val_loss = train_epoch(model, optimizer, criterion, train_loader, validation_loader)
        val_losses.append(val_loss)
    test_loss = evaluate_accuracy(model, test_loder, criterion)

    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optimizer,
        values_to_save={
            "epochs_trained": epochs,
        }
    )
    end = time.time()
    print(f'====> Epoch: {epoch} Average loss: {np.mean(val_losses):.4f}')
    learning_curves, min_valid_seen, min_test_seen = process_trajectory(pipeline_directory, val_loss, test_loss)
    # random search - no fidelity hyperparameter

    if learning_curves["fidelity"] is None:
        return {
        "loss": val_loss,
        "info_dict": {
            "test_loss": test_loss,
            "val_losses": val_losses,
            "train_time": end - start,
            "cost": epochs - start_epoch,
        },
        "cost": epochs - start_epoch,
    }

    return {
        "loss": val_loss,             # validation loss 
        "cost": epochs - start_epoch,
        "info_dict": {
            "cost": epochs - start_epoch,
            "val_score": -val_loss,  # - validation loss for this fidelity
            "test_score": test_loss, # test loss (w/out minus)
            "fidelity": epochs,
            "continuation_fidelity": None,   # keep None
            "start_time": start,
            "end_time": end,
            "max_fidelity_loss": None,
            "max_fidelity_cost": None,
            "min_valid_seen": min_valid_seen,
            "min_test_seen": min_test_seen,
            # "min_valid_ever": None,       # Cannot calculate for real datasets
            # "min_test_ever": None,          # Cannot calculate for real datasets
            "learning_curve": val_loss, # validation loss (w/out minus)
            "learning_curves": learning_curves # dict: valid: [..valid_losses..], test: [..test_losses..], fidelity: [1, 2, ...]
        },
    }

