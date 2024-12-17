import yaml
import torch
import pandas as pd
from pathlib import Path
from pfns_hpo.plot3D import Plotter3D

def load_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"No previous config file found: {file_path}")
        return None

def set_seed(seed=123):
    torch.manual_seed(seed)

def process_trajectory(pipeline_directory, val_loss, val_losses, test_losses, test_loss=None):
    id_fidelity_info = str(pipeline_directory).split("config_")[1]

    if "random_search" in str(pipeline_directory): 
        learning_curve = val_losses

        learning_curves = {
            'fidelity': [i for i in range(1, len(learning_curve) + 1)],
            'valid': val_losses, # single value
            'test': test_losses, # single value

        }
        return learning_curves, val_loss, test_loss

    config_id, fidelity = list(map(int, id_fidelity_info.split("_")))
    previous_results = load_yaml(
        pipeline_directory.parent
        .joinpath(f"config_{config_id}_{fidelity-1}")
        .joinpath('result.yaml')
    )

    if previous_results is None:
        learning_curves = {
            'valid': val_losses,
            'test': test_losses,
            'fidelity': [i for i in range(1, len(val_losses) + 1)],
        }
        return learning_curves, val_loss, test_loss

    learning_curves = previous_results["info_dict"]["learning_curves"]
    # update
    learning_curves['valid'].extend(val_losses)
    if test_losses is not None:
        learning_curves['test'].extend(test_losses)
    learning_curves['fidelity'] = [i for i in range(1, len(learning_curves['valid']) + 1)]
    min_valid_seen = min(val_loss, previous_results["info_dict"].get("min_valid_seen", val_loss))
    min_test_seen = min(test_loss, previous_results["info_dict"].get("min_test_seen", test_loss))
    return learning_curves, min_valid_seen, min_test_seen


def create_3d_plot(
    searcher,
    seed,
    neps_root_directory,
    benchmark,
    normalization_method,
    max_value,
    soft_lb=0.0,
    soft_ub=1.0,
    minimize=False,
    lb=torch.tensor(float("-inf")),
    ub=torch.tensor(float("inf")),
):
    print(f"creating 3D plot for {searcher} on {benchmark} with seed {seed}")

    plotter = Plotter3D(
            algorithm=searcher,
            benchmark=benchmark,
            experiment_group="results_examples",
            seed=seed,
        )

    _df = pd.read_csv(f"{neps_root_directory}/summary_csv/config_data.csv", float_precision="round_trip")
    if normalization_method == "pfn":
        normalize, _ = pfn_normalize(lb, ub, soft_lb, soft_ub, minimize)
    else:
        normalize = neps_normalize(max_value, minimize)
    _df["result.loss"] = normalize(torch.tensor(_df["result.loss"].values)).numpy()
    plotter.plot3D(data=_df, run_path=Path.cwd())


def neps_normalize(max_value, minimize=False):
    return lambda train_y: (
        1 - torch.clamp(train_y, 0, max_value) / max_value
        if minimize
        else torch.clamp(train_y, 0, max_value) / max_value
    )


def pfn_normalize(
    lb=torch.tensor(float("-inf")),
    ub=torch.tensor(float("inf")),
    soft_lb=0.0,
    soft_ub=1.0,
    minimize=False,
):
    """
    LC-PFN curve prior assumes curves to be normalized within the range [0,1] and to be maximized.
    This function allows to normalize and denormalize data to fit this assumption.

    Parameters:
        lb (torch.Tensor): Lower bound of the data.
        ub (torch.Tensor): Upper bound of the data.
        soft_lb (float): Soft lower bound for normalization. Default is 0.0.
        soft_ub (float): Soft upper bound for normalization. Default is 1.0.
        minimize (bool): If True, the original curve is a minization. Default is False.

    Returns: Two functions for normalizing and denormalizing the data.
    """
    assert lb <= soft_lb and soft_lb < soft_ub and soft_ub <= ub
    # step 1: linearly transform [soft_lb,soft_ub] [-1,1] (where the sigmoid behaves approx linearly)
    #    2.0/(soft_ub - soft_lb)*(x - soft_lb) - 1.0
    # step 2: apply a vertically scaled/shifted the sigmoid such that [lb,ub] --> [0,1]

    def cinv(x):
        return 1 - x if minimize else x

    def lin_soft(x):
        return 2 / (soft_ub - soft_lb) * (x - soft_lb) - 1

    def lin_soft_inv(y):
        return (y + 1) / 2 * (soft_ub - soft_lb) + soft_lb

    try:
        if torch.exp(-lin_soft(lb)) > 1e300:
            raise RuntimeError
        # otherwise overflow causes issues, treat these cases as if the lower bound was -infinite
        # print(f"WARNING: {lb} --> NINF to avoid overflows ({np.exp(-lin_soft(lb))})")
    except RuntimeError:
        lb = torch.tensor(float("-inf"))
    if torch.isinf(lb) and torch.isinf(ub):
        return lambda x: cinv(
            1 / (1 + torch.exp(-lin_soft(x)))
        ), lambda y: lin_soft_inv(torch.log(cinv(y) / (1 - cinv(y))))
    elif torch.isinf(lb):
        a = 1 + torch.exp(-lin_soft(ub))
        return lambda x: cinv(
            a / (1 + torch.exp(-lin_soft(x)))
        ), lambda y: lin_soft_inv(torch.log((cinv(y) / a) / (1 - (cinv(y) / a))))
    elif torch.isinf(ub):
        a = 1 / (1 - 1 / (1 + torch.exp(-lin_soft(lb))))
        b = 1 - a
        return lambda x: cinv(
            a / (1 + torch.exp(-lin_soft(x))) + b
        ), lambda y: lin_soft_inv(
            torch.log(((cinv(y) - b) / a) / (1 - ((cinv(y) - b) / a)))
        )
    else:
        a = (
            1
            + torch.exp(-lin_soft(ub))
            + torch.exp(-lin_soft(lb))
            + torch.exp(-lin_soft(ub) - lin_soft(lb))
        ) / (torch.exp(-lin_soft(lb)) - torch.exp(-lin_soft(ub)))
        b = -a / (1 + torch.exp(-lin_soft(lb)))
        return lambda x: cinv(
            a / (1 + torch.exp(-lin_soft(x))) + b
        ), lambda y: lin_soft_inv(
            torch.log(((cinv(y) - b) / a) / (1 - ((cinv(y) - b) / a)))
        )


def check_remaining_configs(searcher, benchmark, seed):
    configs_path = (
        Path("results_examples")
        / f"benchmark={benchmark}"
        / f"algorithm={searcher}"
        / f"seed={seed}"
        / "neps_root_directory"
        / "summary_csv"
        / "config_data.csv"
    )
    results_file = Path.cwd() / f"{searcher}_{seed}_learning_curves_all_configs.csv"

    # load configs
    configs = pd.read_csv(configs_path)
    configs['Config_group'] = configs['Config_id'].apply(lambda x: x.split("_")[0])
    unique_configs = configs.groupby('Config_group').first().reset_index()
    print(f"unique configs shape {unique_configs.shape}")

    # Check for existing results to resume from
    if results_file.exists():
        print("Resuming from existing results file.")
        df_results_existing = pd.read_csv(results_file)
        group_counts = df_results_existing['Config_group'].value_counts()
        completed_groups = set(group_counts[group_counts == 50].index)
        print(f"Completed Config Groups: {completed_groups}")
        incomplete_groups = set(group_counts[group_counts < 50].index)
        if incomplete_groups:
            print(f"Deleting logs for incomplete Config Groups: {incomplete_groups}")
            df_results_existing = df_results_existing[~df_results_existing['Config_group'].isin(incomplete_groups)]
        print(f"Remaining Config Groups after cleanup: {df_results_existing['Config_group'].unique()}")
    else:
        df_results_existing = pd.DataFrame()
        completed_groups = set()

    unique_configs['Config_group'] = unique_configs['Config_group'].astype(str)
    completed_groups = set(map(str, completed_groups))
    remaining_configs = unique_configs[~unique_configs['Config_group'].isin(completed_groups)]
    print(f"Remaining configs shape: {remaining_configs.shape}")

    _df = remaining_configs[['Config_group', 'config.beta1', 'config.beta2', 'config.learning_rate', 'config.epsilon']]
    return _df, results_file


def save_pd_configs(config_group, ep, beta1, beta2, learning_rate, epsilon, val_loss, test_loss, results_file):
    new_result = {
        'Config_group': config_group,
        'epoch': ep + 1,  # Track current epoch
        'config.beta1': beta1,
        'config.beta2': beta2,
        'config.learning_rate': learning_rate,
        'config.epsilon': epsilon,
        'val_loss': val_loss,
        'test_loss': test_loss
    }
    pd.DataFrame([new_result]).to_csv(results_file, mode='a', header=not results_file.exists(), index=False)
