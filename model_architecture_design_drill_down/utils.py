import os

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def compute_accuracy(target, output):
    prediction = torch.argmax(output, dim=1)
    class_predictions = prediction.eq(target).sum().item()
    return class_predictions


def train(model, device, train_loader, optimizer, loss_function):
    model.to(device)
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    data_count = 0

    for batch_index, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = loss_function(output, target)
        correct += compute_accuracy(target, output)
        data_count += len(data)

        accuracy = 100 * correct / data_count

        train_loss += loss.item()
        train_loss = train_loss / len(data)

        loss.backward()
        optimizer.step()

        pbar.set_description(
            f"Training: Train Aggregate Loss: {train_loss:8f}, Batch ID: {batch_index}, Train Accuracy: {accuracy:4f}"
        )


def test(model, device, test_loader, loss_function):
    model.to(device)
    model.eval()

    pbar = tqdm(test_loader)

    test_loss = 0
    correct = 0
    data_count = 0

    for batch_index, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = loss_function(output, target)

        correct += compute_accuracy(target, output)
        data_count += len(data)

        accuracy = 100 * correct / data_count

        test_loss += loss.item()
        test_loss = test_loss / len(data)

        pbar.set_description(
            f"Evaluation: Test Loss: {test_loss:8f}, Test Accuracy: {accuracy:4f}"
        )

    return test_loss


def plot_images(dataloader):
    batch_data, batch_label = next(iter(dataloader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

    if not os.path.exists("images"):
        os.mkdir("images")

    fig.savefig("images/sample_images.png")


def get_device():
    device = "cpu"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    return device
