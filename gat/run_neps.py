import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import tarfile
import argparse
import logging
import neps
from functools import partial
from neps_utils import run_pipeline, get_pipeline_space
from neps_global_utils import set_seed

def main(args):

    # Load the dataset
    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    path = './cora'

    if os.path.isfile(os.path.join(path, 'cora.content')) and os.path.isfile(os.path.join(path, 'cora.cites')):
        print('Dataset already downloaded...')
    else:
        print('Downloading dataset...')
        with requests.get(cora_url, stream=True) as tgz_file:
            with tarfile.open(fileobj=tgz_file.raw, mode='r:gz') as tgz_object:
                tgz_object.extractall()
    
    set_seed(args.seed)
    pipeline_space = get_pipeline_space(args.searcher)
    logging.basicConfig(level=logging.INFO)
    
    run_pipeline_partial = partial(run_pipeline, n_hidden=args.n_hidden, dropout=args.dropout, leaky_relu_slope=args.leaky_relu_slope, concat_heads=args.concat_heads, n_heads=args.n_heads)

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
    parser = argparse.ArgumentParser(description="gat ifbo")
    parser.add_argument("--searcher", type=str, default="ifbo")
    parser.add_argument("--max_cost_total", type=int, default=50)
    parser.add_argument("--overwrite_working_directory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--searcher_path", type=str, default="/home/theo/development/automl/ta_forks/tj-ifbo_private/src/pfns_hpo/pfns_hpo/configs/algorithm")
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--leaky_relu_slope", type=float, default=0.2)
    parser.add_argument("--concat_heads", action="store_true", default=False)
    parser.add_argument("--n_heads", type=int, default=8)
    args = parser.parse_args()
    main(args)