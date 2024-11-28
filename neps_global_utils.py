import yaml
import torch
import neps

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed=123):
    torch.manual_seed(seed)

def process_trajectory(pipeline_directory, val_loss, test_loss):
    id_fidelity_info = str(pipeline_directory).split("config_")[1]
    
    if "random_search" in str(pipeline_directory) or "hyperband" in str(pipeline_directory): ## random search - no fidelity hyperparameter
        l_curves = {
            'valid': [val_loss,], # single value
            'test': [test_loss,], # single value
            'fidelity': None
        }
        return l_curves, val_loss, test_loss
    config_id, fidelity = list(map(int, id_fidelity_info.split("_")))
    # load the results of the previous fidelity for this configuration
    if fidelity == 0:
        l_curves = {
            'valid': [val_loss,],
            'test': [test_loss,],
            'fidelity': [1,],
        }
        return l_curves, val_loss, test_loss
    previous_results = load_yaml(
        pipeline_directory.parent
        .joinpath(f"config_{config_id}_{fidelity-1}")
        .joinpath('result.yaml')
    )
    l_curves = previous_results["info_dict"]["learning_curves"]
    # update
    l_curves['valid'].append(val_loss)
    l_curves['test'].append(test_loss)
    # increment by 1
    l_curves['fidelity'].append(l_curves['fidelity'][-1] + 1)
    min_valid_seen = min(val_loss, previous_results["info_dict"]["min_valid_seen"])
    min_test_seen = min(test_loss, previous_results["info_dict"]["min_test_seen"])
    return l_curves, min_valid_seen, min_test_seen


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