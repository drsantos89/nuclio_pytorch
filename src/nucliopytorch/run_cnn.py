"""Run a feed forward neural network on the MNIST dataset."""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from nucliopytorch.load.get_data import download_data_fashion, get_data_loader
from nucliopytorch.model.pytorch_cnn import TorchCNN

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main(epochs: int = 5) -> None:
    """Run main function."""
    # load training and test data
    train_dataset, test_dataset = download_data_fashion()

    # create data loaders
    train_loader = get_data_loader(train_dataset, 100, True)
    test_loader = get_data_loader(test_dataset, 100, False)

    # initialize model
    model = TorchCNN(
        input_channels=1,
        output_channels=10,
        kernel_size=3,
        stride=1,
        dropout=0.2,
        activation="relu",
        output_size=10,
    )

    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize optimizer, loss function and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    # initialize lists to store training and test loss
    train_loss = []
    test_loss = []

    # initialize variables to store best model and patience for early stopping
    best_model: float = 9.0
    best_model_patience: int = 0

    # train model
    for epoch in range(epochs):
        # set model to training mode
        model.train()
        # initialize loss sum
        loss_sum = 0
        # iterate over training data
        for batch_idx, (data, target) in enumerate(train_loader):  # noqa: B007
            # move data to GPU if available
            data = data.to(device)
            target = target.to(device)
            # zero out gradients
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # calculate loss
            loss = loss_fn(output, target)
            # add loss to loss sum
            loss_sum += loss.item()
            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target).sum().item() / target.size(0)
            # backpropagate
            loss.backward()
            # update weights
            optimizer.step()
        print(
            f"Train -> Epoch: {epoch}, Loss: {loss_sum/batch_idx}, Accuracy: {accuracy}"
        )
        # append loss to list
        train_loss.append(loss_sum / batch_idx)

        # set model to evaluation mode
        model.eval()
        loss_sum = 0
        # set torch.no_grad() because we don't need to calculate gradients
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):  # noqa: B007
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                loss_sum += loss.item()
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item() / target.size(0)
        # update learning rate
        scheduler.step(loss_sum / batch_idx)
        # save best model and early stop if patience is reached
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

    # plot loss
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
