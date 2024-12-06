import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
import neps
from model import TransformerModel, RNNModel
from neps.utils.common import load_checkpoint, save_checkpoint
from neps_global_utils import process_trajectory

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


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model, criterion, data, ntokens, eval_batch_size, bptt):
    model.eval()
    val_loss = 0.0
    if not isinstance(model, TransformerModel):
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, bptt):
            input, targets = get_batch(data, i, bptt)
            if isinstance(model, TransformerModel):
                output = model(input)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(input, hidden)
                hidden = repackage_hidden(hidden)
            val_loss += len(input) * criterion(output, targets).item()
    return val_loss / (len(data) - 1)


def train_epoch(
    model,
    optimizer,
    criterion,
    ntokens,
    train_data,
    val_data,
    eval_batch_size,
    batch_size,
    bptt,
    clip,
):
    model.train()

    if not isinstance(model, TransformerModel):
        hidden = model.init_hidden(batch_size)

    for _, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        if isinstance(model, TransformerModel):
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    val_loss = evaluate(model, criterion, val_data, ntokens, eval_batch_size, bptt)

    return val_loss


def run_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    epoch=50,  # 50 default if not handled by the searcher
    opts=None,
    corpus=None,
    eval_batch_size=10,
):
    start = time.time()
    epochs = int(epoch)

    criterion = torch.nn.NLLLoss()

    ntokens = len(corpus.dictionary)
    if opts.model == "Transformer":
        model = TransformerModel(
            ntokens, opts.emsize, opts.nhead, opts.nhid, opts.nlayers, opts.dropout
        ).to(device)
    else:
        model = RNNModel(
            opts.model,
            ntokens,
            opts.emsize,
            opts.nhid,
            opts.nlayers,
            opts.dropout,
            opts.tied,
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon
    )

    previous_state = load_checkpoint(
        directory=previous_pipeline_directory, model=model, optimizer=optimizer
    )

    start_epoch = previous_state["epochs_trained"] if previous_state is not None else 0

    train_data = batchify(corpus.train, opts.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    val_losses, test_losses = [], []

    for ep in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(ep + 1, epochs).ljust(2))
        val_loss = train_epoch(
            model,
            optimizer,
            criterion,
            ntokens,
            train_data,
            val_data,
            eval_batch_size,
            opts.batch_size,
            opts.bptt,
            opts.clip,
        )
        val_losses.append(val_loss)
        test_loss = evaluate(
            model, criterion, test_data, ntokens, eval_batch_size, opts.bptt
        )
        test_losses.append(test_loss)

    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optimizer,
        values_to_save={
            "epochs_trained": epochs,
        },
    )

    end = time.time()

    print(f"  Epoch {epochs} / {epochs} Val Loss: {val_loss}".ljust(2))
    
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
