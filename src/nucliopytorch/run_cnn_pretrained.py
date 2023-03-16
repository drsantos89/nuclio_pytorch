"""Run a feed forward neural network on the MNIST dataset."""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from nucliopytorch.load.get_data import download_data_fashion, get_data_loader
from nucliopytorch.model.pytorch_cnn import get_efficientnet

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main(epochs: int = 5) -> None:
    """Run main function."""
    train_dataset, test_dataset = download_data_fashion()

    train_loader = get_data_loader(train_dataset, 100, True)

    test_loader = get_data_loader(test_dataset, 100, False)

    model = get_efficientnet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    train_loss = []
    test_loss = []
    best_model: float = 9.0
    best_model_patience: int = 0

    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):  # noqa: B007
            data = data.to(device)
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
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                loss_sum += loss.item()
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item() / target.size(0)
        scheduler.step(loss_sum / batch_idx)
        if loss_sum / batch_idx < best_model:
            best_model = loss_sum / batch_idx
            torch.save(model.state_dict(), "./results/best_model.pt")
            best_model_patience = 0
        else:
            best_model_patience += 1
            if best_model_patience > 4:
                break
        print(
            f"Test -> Epoch: {epoch}, Loss: {loss_sum/batch_idx}, Accuracy: {accuracy}"
        )
        test_loss.append(loss_sum / batch_idx)

    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
