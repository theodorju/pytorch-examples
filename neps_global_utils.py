import yaml
import torch
import neps

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
    learning_curves['test'].extend(test_losses)
    learning_curves['fidelity'] = [i for i in range(1, len(learning_curves['valid']) + 1)]
    min_valid_seen = min(val_loss, previous_results["info_dict"].get("min_valid_seen", val_loss))
    min_test_seen = min(test_loss, previous_results["info_dict"].get("min_test_seen", test_loss))
    return learning_curves, min_valid_seen, min_test_seen
