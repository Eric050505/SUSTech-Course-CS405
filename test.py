from torch import nn, optim
from tqdm import tqdm

from detection import TrafficSignDetectionCNN
from dataset import *
from util import *
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()  # Converts PIL image to tensor and scales values to [0, 1]
])

# Example usage:
img_dir = './mini_test/images'
label_dir = './mini_test/labels'
dataset = TrafficSignDataset(img_dir, label_dir, transform=transform)
train_loader, test_loader = split_dataset(dataset, train_ratio=0.8, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = TrafficSignDetectionCNN()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for imgs, boxes, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'Loss': loss.item()})
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {running_loss / len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for imgs, boxes, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")
