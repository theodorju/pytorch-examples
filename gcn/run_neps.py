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
    
    run_pipeline_partial = partial(run_pipeline, hidden_dim=args.hidden_dim, dropout_p=args.dropout_p, include_bias=args.include_bias)

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
    parser = argparse.ArgumentParser(description="gcn ifbo")
    parser.add_argument("--searcher", type=str, default="ifbo")
    parser.add_argument("--max_cost_total", type=int, default=50)
    parser.add_argument("--overwrite_working_directory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--searcher_path", type=str, default="/home/theo/development/automl/ta_forks/tj-ifbo_private/src/pfns_hpo/pfns_hpo/configs/algorithm")
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--include_bias", action="store_true", default=False)
    args = parser.parse_args()
    main(args)