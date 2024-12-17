import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import neps
import torch
from datetime import date
from functools import partial
from neps_utils import run_pipeline, get_pipeline_space
from neps_global_utils import set_seed, create_3d_plot


def main(args):

    set_seed(args.seed)
    pipeline_space = get_pipeline_space(args.searcher)
    logging.basicConfig(level=logging.INFO)

    run_pipeline_partial = partial(run_pipeline, opts=args)

    neps_root_directory = f"results_examples/benchmark=language_translation/algorithm={args.searcher}/seed={args.seed}/neps_root_directory"

    # make directory if necessary
    if not os.path.exists(neps_root_directory):
        os.makedirs(neps_root_directory)
    
    if not args.plot_only:
        neps.run(
            run_pipeline=run_pipeline_partial,
            pipeline_space=pipeline_space,
            root_directory=neps_root_directory,
            overwrite_working_directory=args.overwrite_working_directory,
            max_cost_total=args.max_cost_total,
            searcher=args.searcher,
            searcher_path=args.searcher_path,
            post_run_summary=True,
            surrogate_model_args={
                # 'soft_ub': 9.827901565579532, # np.log(tgt_vocab_size=18544)
                # 'soft_lb': 0.0,
                # 'lb': 0.0,
                # 'already_normalized': False,
                "normalization_method": "neps",
                "max_value": 10, # bit higher that np.log(tgt_vocab_size=18544)
            },
        )

    if "ifbo" in args.searcher: # includes any ifbo variant
        create_3d_plot(
            args.searcher,
            args.seed,
            neps_root_directory,
            benchmark="language_translation",
            normalization_method="neps",
            soft_lb=torch.tensor(0.0),
            soft_ub=torch.tensor(9.827901565579532),
            lb=torch.tensor(0.0),
            max_value=10,
            minimize=False,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="language translation ifbo")
    parser.add_argument("--searcher", type=str, default="ifbo")
    parser.add_argument("--max_cost_total", type=int, default=50)
    parser.add_argument("--overwrite_working_directory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--searcher_path",
        type=str,
        default="/home/theo/development/automl/ta_forks/tj-ifbo_private/src/pfns_hpo/pfns_hpo/configs/algorithm",
    )
    # Translation settings
    parser.add_argument(
        "--src",
        type=str,
        default="de",
        help="Source language (translating FROM this language)",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default="en",
        help="Target language (translating TO this language)",
    )
    # Training settings
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=5, help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=5, help="Number of decoder layers")
    parser.add_argument("--embed_size", type=int, default=512, help="Size of the language embedding")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Feedforward dimensionality")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout")
    # Logging settings
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./" + str(date.today()) + "/",
        help="Where the output of this program should be placed",
    )
    # Just cause it's used in the pre-existing code
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()
    main(args)
