import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import neps
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from neps.utils.common import load_checkpoint, save_checkpoint
from neps_global_utils import set_seed, process_trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pipeline_space(searcher) -> dict:  # maybe limiting for ifbo
    """define search space for neps"""
    pipeline_space = dict(
        learning_rate=neps.FloatParameter(
            lower=1e-9,
            upper=10,
            log=True,
        ),
        beta1=neps.FloatParameter(
            lower=1e-4,
            upper=1,
            log=True,
        ),
        beta2=neps.FloatParameter(
            lower=1e-3,
            upper=1,
            log=True,
        ),
        epsilon=neps.FloatParameter(
            lower=1e-12,
            upper=1000,
            log=True,
        )
    )
    uses_fidelity = ("ifbo", "hyperband", "asha", "ifbo_taskset_4p", "ifbo_taskset_4p_extended")
    if searcher in uses_fidelity:
        pipeline_space["epoch"] = neps.IntegerParameter(
            lower=1,
            upper=50,
            is_fidelity=True,
        )
    return pipeline_space


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # nan --> diverged
    if torch.isnan(recon_x).any().item():
        return float('inf')
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

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

def evaluate_accuracy(model, data_loader, criterion):
    set_seed()
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss += criterion(recon_batch, data, mu, logvar).item() * data.size(0)
    loss /= len(data_loader.dataset)
    return loss

def load_mnist(batch_size, valid_size, val_test_batch_size=1024):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="../data", train=False, transform=transform)

    train_dataset, valid_dataset = random_split(dataset, [1-valid_size, valid_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset=valid_dataset, batch_size=val_test_batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=val_test_batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


def adaptive_gradient_clipping(model, clip_factor=2, percentile=95, max_history=100):
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
        if loss == float('inf'):
            return float('inf')
        loss.backward()
        adaptive_gradient_clipping(model, clip_factor=2)
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
        epoch=50 # 50 default if not handled by the searcher
):
    start = time.time()
    # for mf algorithms
    epochs = int(epoch)

    criterion = loss_function
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)

    train_loader, validation_loader, test_loder = load_mnist(batch_size=128, valid_size=0.2)

    previous_state = load_checkpoint(
        directory=previous_pipeline_directory,
        model=model,
        optimizer=optimizer
    )

    start_epoch = previous_state["epochs_trained"] if previous_state is not None else 0
    
    val_losses, test_losses = [], []

    for ep in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(ep + 1, epochs).ljust(2))
        val_loss = train_epoch(model, optimizer, criterion, train_loader, validation_loader)
        val_losses.append(val_loss)
    
        if val_loss == float('inf'):
            test_loss = float('inf')
            test_losses.append(test_loss)
        else:
            test_loss = evaluate_accuracy(model, test_loder, criterion)
            test_losses.append(test_loss)

    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optimizer,
        values_to_save={
            "epochs_trained": epochs,
        }
    )
    end = time.time()

    if val_loss == float('inf'):
        print(f"====> Diverged")
    else:
        print(f'====> Epoch: {epoch} Val loss: {val_loss}')
    
    learning_curves, min_valid_seen, min_test_seen = process_trajectory(
        pipeline_directory, val_loss, val_losses, test_losses, test_loss
    )

    return {
        "cost": epochs - start_epoch,
        "info_dict": {
            "continuation_fidelity": None,
            "cost": epochs - start_epoch,
            "end_time": end,
            "fidelity": epochs,
            "learning_curve": val_losses,
            "learning_curves": learning_curves,
            "max_fidelity_cost": epochs,
            "max_fidelity_loss": val_losses[-1],
            # "min_test_ever": np.min(test_losses),
            "min_test_seen": np.min(learning_curves["test"]),
            # "min_valid_ever": np.min(val_losses),
            "min_valid_seen": np.min(learning_curves["valid"]),
            "process_id": os.getpid(),
            "start_time": start,
            "test_score": test_loss,
            "val_score": -val_loss,
        },
        "loss": val_loss,
    }
