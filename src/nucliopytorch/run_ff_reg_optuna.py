"""Run a feed forward neural network on the MNIST dataset."""
import json
import random

import numpy as np
import optuna
import torch

from nucliopytorch.load.get_data import download_data, get_data_loader
from nucliopytorch.model.pytorch_ff import TorchFF

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def objective(trial: optuna.trial.Trial) -> float:
    """Run main function."""
    train_dataset, test_dataset = download_data()

    train_loader = get_data_loader(train_dataset, 100, True)

    test_loader = get_data_loader(test_dataset, 100, False)

    model = TorchFF(
        input_size=784,
        hidden_size=trial.suggest_int("hidden_size", 32, 512, log=True),
        output_size=10,
        dropout=trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    train_loss = []
    test_loss = []
    best_model: float = 9.0
    best_model_patience: int = 0

    for epoch in range(1):
        model.train()
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):  # noqa: B007
            data = data.reshape(-1, 28 * 28).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss_sum += loss.item()
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target).sum().item() / target.size(0)
            loss.backward()
            optimizer.step()
        print(
            f"Train -> Epoch: {epoch}, Loss: {loss_sum/batch_idx}, Accuracy: {accuracy}"
        )
        train_loss.append(loss_sum / batch_idx)

        model.eval()
        loss_sum = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):  # noqa: B007
                data = data.reshape(-1, 28 * 28).to(device)
                target = target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                loss_sum += loss.item()
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item() / target.size(0)
        scheduler.step(loss_sum / batch_idx)
        if loss_sum / batch_idx < best_model:
            best_model = loss_sum / batch_idx
            torch.save(
                model.state_dict(),
                f"./results/{trial.study.study_name}_{trial.number}.pt",
            )
            best_model_patience = 0
        else:
            best_model_patience += 1
            if best_model_patience > 4:
                break
        print(
            f"Test -> Epoch: {epoch}, Loss: {loss_sum/batch_idx}, Accuracy: {accuracy}"
        )
        test_loss.append(loss_sum / batch_idx)

        with open(f"./results/{trial.study.study_name}_{trial.number}.json", "w") as f:
            json.dump([trial.params, train_loss, test_loss], f)

    return loss_sum / batch_idx


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="ff_reg_exp1")
    study.optimize(objective, n_trials=3, n_jobs=1, timeout=600)

    print(study.best_params)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
