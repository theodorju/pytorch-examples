import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from neps_global_utils import check_remaining_configs, save_pd_configs
from neps_utils import Net, load_mnist, train_epoch, evaluate_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

def train_configs(searcher, seed, epochs):
    df, results_file = check_remaining_configs(searcher, "mnist", seed)
    train_loader, validation_loader, test_loader = load_mnist(batch_size=64, valid_size=0.2)

    for _, row in df.iterrows():
        config_group = row['Config_group']
        beta1 = row['config.beta1']
        beta2 = row['config.beta2']
        learning_rate = row['config.learning_rate']
        epsilon = row['config.epsilon']
        print(f"Training config {config_group} with beta1={beta1}, beta2={beta2}, learning_rate={learning_rate}, epsilon={epsilon}")

        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        criterion = torch.nn.NLLLoss()

        for ep in range(epochs):
            _, _, val_loss = train_epoch(model, optimizer, criterion, train_loader, validation_loader)
            _, _, test_loss = evaluate_accuracy(model, test_loader, criterion)

            print(f"  Epoch {ep + 1} / {epochs} Val Loss: {val_loss}".ljust(2))
            save_pd_configs(config_group, ep, beta1, beta2, learning_rate, epsilon, val_loss, test_loss, results_file)

    print("All configurations processed. Final results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--searcher", type=str, default="ifbo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_configs(args.searcher, args.seed, args.epochs)
