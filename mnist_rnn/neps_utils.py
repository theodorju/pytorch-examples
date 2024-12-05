import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import neps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neps.utils.common import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader, random_split
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, input):
        # Shape of input is (batch_size,1, 28, 28)
        # converting shape of input to (batch_size, 28, 28)
        # as required by RNN when batch_first is set True
        input = input.reshape(-1, 28, 28)
        output, hidden = self.rnn(input)

        # RNN output shape is (seq_len, batch, input_size)
        # Get last output of RNN
        output = output[:, -1, :]
        output = self.batchnorm(output)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        output = F.log_softmax(output, dim=1)
        return output

def train_epoch(model, optimizer, criterion, train_loader, validation_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    val_acc, val_err, val_loss = evaluate_accuracy(model, validation_loader, criterion)

    return val_acc, val_err, val_loss

def evaluate_accuracy(model, data_loader, criterion):
    set_seed()
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            loss += criterion(output, target).item() * data.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(data_loader.dataset)
    error = 1 - accuracy
    loss /= len(data_loader.dataset)
    return accuracy, error, loss


def run_pipeline(
        pipeline_directory,
        previous_pipeline_directory,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        epoch=50, # 14 default if not handled by the searcher
):
    start = time.time()
    # for mf algorithms
    epochs = int(epoch)

    criterion = torch.nn.NLLLoss()
    model = Net().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon
    )

    train_loader, validation_loader, test_loader = load_mnist(batch_size=64, valid_size=0.2)

    # checkpointing to resume model training in higher fidelities
    previous_state = load_checkpoint(
        directory=previous_pipeline_directory,
        model=model,
        optimizer=optimizer,
    )

    if previous_state is not None:
        start_epoch = previous_state["epochs_trained"]
    else:
        start_epoch = 0

    val_errors = list()

    # train the model
    for ep in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(ep + 1, epochs).ljust(2))
        val_acc, val_error, val_loss = train_epoch(model, optimizer, criterion, train_loader, validation_loader)
        val_errors.append(val_error)
    test_acc, test_error, test_loss = evaluate_accuracy(model, test_loader, criterion)

    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optimizer,
        values_to_save={
            "epochs_trained": epochs,
        }
    )
    print(f"  Epoch {epochs} / {epochs} Val Loss: {val_loss}".ljust(2))
    end = time.time()

    learning_curves, min_valid_seen, min_test_seen = process_trajectory(
        pipeline_directory, val_loss, test_loss
    )
    # random search - no fidelity hyperparameter
    if "random_search" in str(pipeline_directory) or "hyperband" in str(pipeline_directory):
        return {
        "loss": val_loss,
        "info_dict": {
            "test_accuracy": test_acc,
            "val_errors": val_errors,
            "val_loss": val_loss,
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