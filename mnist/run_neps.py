import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import neps
from neps_utils import run_pipeline, get_pipeline_space
from neps_global_utils import set_seed

def main(args):
    set_seed(args.seed)
    pipeline_space = get_pipeline_space(args.searcher)
    logging.basicConfig(level=logging.INFO)

    neps_root_directory = f"results_examples/benchmark=mnist/algorithm={args.searcher}/seed={args.seed}/neps_root_directory"
    # make directory if necessary
    if not os.path.exists(neps_root_directory):
        os.makedirs(neps_root_directory)

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=neps_root_directory,
        overwrite_working_directory=args.overwrite_working_directory,
        max_cost_total=args.max_cost_total,
        searcher=args.searcher,
        searcher_path=args.searcher_path,
        post_run_summary=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mnist ifbo")
    parser.add_argument("--searcher", type=str, default="ifbo")
    parser.add_argument("--max_cost_total", type=int, default=50)
    parser.add_argument("--overwrite_working_directory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--searcher_path", type=str, default="/home/theo/development/automl/ta_forks/tj-ifbo_private/src/pfns_hpo/pfns_hpo/configs/algorithm")
    args = parser.parse_args()
    main(args)