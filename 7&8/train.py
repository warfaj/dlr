# 7 & 8 MNIST classification on vector inputs with SGD or Adam optimizer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import FeedForwardMNISTClassifier
import matplotlib.pyplot as plt

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
epochs = 5
learning_rate = 0.001
max_iters = 1000
eval_iter = 100
optimizer = "sgd" # sgd or adam


# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = FeedForwardMNISTClassifier().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(X,y, model, loss_fn, optimizer, training_losses):
    model.train()
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    training_losses.append(loss.item())

def test(dataloader, model, loss_fn, test_losses):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    test_losses.append(test_loss.item())
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 



training_losses, test_losses = [],[]
iters = 0
test_iters = []
train_iter = iter(train_dataloader)

while iters < max_iters:
    try:
        X, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        X, y = next(train_iter)

    train(X,y, model, loss_fn, optimizer, training_losses)

    if iters % eval_iter == 0:
        test_iters.append(iters)
        test(test_dataloader, model, loss_fn, test_losses)

    iters += 1

print("Final Test Set Evaluation")
test_iters.append(iters)
test(test_dataloader, model, loss_fn, test_losses)


# Plot loss curve
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(training_losses)
plt.title('Train Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.savefig("training_graph.png", bbox_inches="tight")


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(test_iters, test_losses)
plt.title('Test Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.savefig("test_graph.png", bbox_inches="tight")
