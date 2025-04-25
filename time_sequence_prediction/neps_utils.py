from __future__ import print_function
import math
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from neps.utils.common import load_checkpoint, save_checkpoint
from neps_global_utils import process_trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


def evaluate_accuracy(model, test_input, test_target, criterion):
    model.eval()
    with torch.no_grad():
        future = 1000
        pred = model(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        return loss

def train_epoch(model, optimizer, criterion, input, target, test_input, test_target, n_params=8, l1=None, l2=None):
    model.train()
    optimizer.zero_grad()
    out = model(input)
    loss = criterion(out, target)
    if n_params == 8:
            l1_loss = 0
            l2_loss = 0
            # apply l1 and l2 regularization
            for p in model.parameters():
                l1_loss += torch.sum(torch.abs(p))
                l2_loss += torch.sum(p ** 2)
            loss += l1 * l1_loss + l2 * l2_loss
    loss.backward()
    optimizer.step()
    val_loss = evaluate_accuracy(model, test_input, test_target, criterion)
    return val_loss


def run_pipeline(
        pipeline_directory,
        previous_pipeline_directory,
        learning_rate,
        beta1=None,
        beta2=None,
        epsilon=None,
        l1=None,
        l2=None,
        linear_decay=None,
        exponential_decay=None,
        epoch=50, # 50 default if not handled by the searcher
        n_params=4,
):
    start = time.time()
    # for mf algorithms
    epochs = int(epoch)
    criterion = nn.MSELoss()
    model = Sequence().to(device)
    model.double()

    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    if n_params == 1:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon
        )

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

    val_losses, test_losses = [], []
    for ep in range(start_epoch, epochs):
        if n_params == 8:
            linear_factor = np.max(1 - linear_decay * ep, 0)
            exponential_factor = np.exp(-exponential_decay * ep)
            updated_lr = learning_rate * linear_factor * exponential_factor
            # update lr manually
            assert len(optimizer.param_groups) == 1
            optimizer.param_groups[0]["lr"] = updated_lr
        
        val_loss = train_epoch(model, optimizer, criterion, input, target, test_input, test_target, n_params, l1, l2)

        if math.isnan(val_loss):
            val_loss = float('inf')
        val_losses.append(val_loss)
    
    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optimizer,
        values_to_save={
            "epochs_trained": epochs,
        }
    )
    end = time.time()
    learning_curves, min_valid_seen, min_test_seen = process_trajectory(
        pipeline_directory, val_loss, val_losses, test_losses=None, test_loss=None
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
            "test_score": None,
            "val_score": -val_loss,
        },
        "loss": val_loss,
    }