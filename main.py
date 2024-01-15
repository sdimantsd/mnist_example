import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(5 * 5 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 5 * 5 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train_epoch(model, train_loader, optimizer, criterion, epoch, losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        percentage = 100. * batch_idx / len(train_loader)
        print(f'\rEpoch: {epoch} | Progress: {percentage:.2f}% | Loss: {loss.item():.6f}', end='')

    return losses


# Main training loop
def train(model, train_loader, optimizer, criterion, epochs=2, save_weights_path="./model_weights.pth"):
    losses = []
    for epoch in range(epochs):
        losses = train_epoch(model, train_loader, optimizer, criterion, epoch, losses)
    print()  # New line after training

    # Save the model weights after training
    torch.save(model.state_dict(), save_weights_path)

def val(model, val_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'\nValidation set: Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.0f}%)\n')

    # Display some images with their predictions
    fig = plt.figure(figsize=(12, 6))
    for i in range(6):
        ax = fig.add_subplot(1, 6, i + 1)
        ax.imshow(data[i][0].cpu().numpy().squeeze(), cmap='gray', interpolation='none')
        pred_label = output.data.max(1, keepdim=True)[1][i].item()
        color = 'green' if pred_label == target[i].item() else 'red'
        ax.set_title(f"Pred: {pred_label}", color=color, fontsize=9)
        ax.axis('off')
    plt.tight_layout(pad=1.0)
    plt.show()

# Settings
batch_size = 64
lr = 0.01
momentum = 0.9

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                                          batch_size=batch_size, shuffle=True)

# Model, optimizer, and loss function setup
model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss()

# Training
train(model, train_loader, optimizer, criterion)

model_weights_path = "./model_weights.pth"
model.load_state_dict(torch.load(model_weights_path))
# Validation
val(model, test_loader)

