import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import neps
import data
from neps_utils import run_pipeline
from neps_global_utils import set_seed, get_pipeline_space
from functools import partial

def main(args):
    set_seed(args.seed)
    pipeline_space = get_pipeline_space(args.searcher)
    logging.basicConfig(level=logging.INFO)
    corpus = data.Corpus(args.data)

    run_pipeline_partial = partial(run_pipeline, opts=args, corpus=corpus, eval_batch_size=args.eval_batch_size)

    # ifbo
    neps.run(
        run_pipeline=run_pipeline_partial,
        pipeline_space=pipeline_space,
        root_directory=f"results_examples/{args.searcher}_seed={args.seed}",
        overwrite_working_directory=args.overwrite_working_directory,
        max_cost_total=args.max_cost_total,
        searcher=args.searcher,
        searcher_path=args.searcher_path,
        post_run_summary=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="word language model ifbo")
    parser.add_argument("--searcher", type=str, default="ifbo")
    parser.add_argument("--max_cost_total", type=int, default=50)
    parser.add_argument("--overwrite_working_directory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--searcher_path",
        type=str,
        default="/home/theo/development/automl/ta_forks/tj-ifbo_private/src/pfns_hpo/pfns_hpo/configs/algorithm",
    )

    # example specific arguments
    parser.add_argument(
        "--data",
        type=str,
        default="./data/wikitext-2",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LSTM",
        help="type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)",
    )
    parser.add_argument(
        "--emsize", type=int, default=200, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=200, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=10,
    )
    parser.add_argument("--bptt", type=int, default=35, help="sequence length")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument(
        "--tied", action="store_true", help="tie the word embedding and softmax weights"
    )
    # parser.add_argument("--cuda", action="store_true", default=False, help="use CUDA")
    # parser.add_argument(
    #     "--mps", action="store_true", default=False, help="enables macOS GPU training"
    # )
    # parser.add_argument(
    #     "--log-interval", type=int, default=200, metavar="N", help="report interval"
    # )
    # parser.add_argument(
    #     "--save", type=str, default="model.pt", help="path to save the final model"
    # )
    # parser.add_argument(
    #     "--onnx-export",
    #     type=str,
    #     default="",
    #     help="path to export the final model in onnx format",
    # )
    parser.add_argument(
        "--nhead",
        type=int,
        default=2,
        help="the number of heads in the encoder/decoder of the transformer model",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="verify the code and the model"
    )

    args = parser.parse_args()
    main(args)
