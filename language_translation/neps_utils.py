import time
import os
import numpy as np
import torch
import neps
from main import get_data, Translator, train, validate
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


def run_pipeline(
        pipeline_directory,
        previous_pipeline_directory,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        epoch=50,  # 30 default if not handled by the searcher
        opts=None,
):
    start = time.time()
    epochs = int(epoch)

    train_dl, valid_dl, src_vocab, tgt_vocab, _, _, special_symbols = get_data(opts)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model = model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    previous_state = load_checkpoint(
        directory=previous_pipeline_directory,
        model=model,
        optimizer=optimizer,
    )

    if previous_state is not None:
        start_epoch = previous_state["epochs_trained"]
    else:
        start_epoch = 0

    val_losses = []

    for idx, ep in enumerate(range(start_epoch, epochs)):
        print("  Epoch {} / {} ...".format(ep + 1, epochs).ljust(2))
        train_loss = train(model, train_dl, loss_fn, optimizer, special_symbols, opts)
        val_loss = validate(model, valid_dl, loss_fn, special_symbols)
        val_losses.append(val_loss)


    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optimizer,
        values_to_save={
            "epochs_trained": epochs,
        },
    )
    print(f"  Epoch {epochs} / {epochs} Val Loss: {val_loss}".ljust(2))
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
            "test_score": test_loss,
            "val_score": -val_loss,
        },
        "loss": val_loss,
    }