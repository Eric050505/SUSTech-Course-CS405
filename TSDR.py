import torch
from tqdm import tqdm

from data_loader import TrafficSignDataset
from recognizer import RecognizerCNN
from detector import Detector


class TSDR:
    # Traffic Signals Detection and Recognition
    def __init__(self, data_loader: TrafficSignDataset, detector: Detector, recognizer: RecognizerCNN):
        self.data_loader = data_loader
        self.detector = detector
        self.recognizer = recognizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_R(self):
        num_epochs = 10
        for epoch in range(num_epochs):
            self.recognizer.train()
            running_loss = 0.0
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for imgs, boxes, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.recognizer.optimizer.zero_grad()
                outputs = self.recognizer(imgs)
                loss = self.recognizer.criterion(outputs, labels)
                loss.backward()
                self.recognizer.optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({'Loss': loss.item()})
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {running_loss / len(self.recognizer.train_loader):.4f}")

    def pred_R(self, test_loader: TrafficSignDataset):
        self.recognizer.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for imgs, boxes, labels in test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.recognizer(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Accuracy: {100 * correct / total:.2f}%")
