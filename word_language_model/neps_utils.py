import time
import torch
import model as m
import neps
from neps.utils.common import load_checkpoint, save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, model_type, corpus):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # model.zero_grad()
        optimizer.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

def get_pipeline_space(searcher) -> dict:
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
    uses_fidelity = ("ifbo", "hyperband", "asha")
    if searcher in uses_fidelity:
        pipeline_space["epoch"] = neps.IntegerParameter(
            lower=1,
            upper=14,
            is_fidelity=True,
        )
    return pipeline_space


def run_pipeline(
        model_type,
        ntokens,
        emsize,
        nhead,
        nhid,
        nlayers,
        dropout,
        tied,
        pipeline_directory,
        previous_pipeline_directory,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        epoch=40  # 40 default if not handled by the searcher
):
    start = time.time()
    epochs = int(epoch)
    criterion = torch.nn.NLLLoss()
    if model_type == 'transformer':
        model = m.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    else:
        model = m.RNNModel(model_type, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)
    optimizer = torch.optim.Adam(model.paramters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)

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
    for ep in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(ep + 1, epochs).ljust(2))
        