import yaml
import torch
import neps

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed=123):
    torch.manual_seed(seed)

def process_trajectory(pipeline_directory, val_loss, test_loss=None):
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
