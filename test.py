import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchvision import transforms
from data_loader import *
from util import *
from detector import *


def main():
    # Step 1: Load dataset
    image_dir = './mini_test/images'  # Image folder path
    label_dir = './mini_test/labels'  # Label folder path

    # Define any image transformations you want to apply
    transform = transforms.Compose([
        transforms.Resize((2048, 2048)),  # Optional resize
        transforms.ToTensor()  # Convert to tensor
    ])

    # Create dataset
    dataset = TrafficSignDataset(image_dir, label_dir, transform=transform)

    # Step 2: Split the dataset into training and testing sets
    train_loader, test_loader = split_dataset(dataset, train_ratio=0.8, batch_size=4, shuffle=True,
                                              collate_fn=collate_fn)

    # Step 3: Prepare training data for SVM
    # For simplicity, we'll use the entire dataset (not the DataLoader, which is for batch loading)
    X_train, y_train = generate_training_data(dataset)

    # Step 4: Train SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    # Step 5: Prepare test data and evaluate the SVM
    X_test, y_test = generate_training_data(test_loader.dataset)  # Assuming test data is handled the same way
    y_pred = svm_classifier.predict(X_test)

    # Print evaluation results
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred))

    # Step 6: Use the trained SVM to detect objects in test images (using sliding window)
    for img, boxes, labels in test_loader:
        img = img[0]  # Take the first image in the batch for demonstration
        detected_boxes = detect_objects(img, svm_classifier)

        # Output the detected boxes (bounding box coordinates)
        print(f"Detected boxes: {detected_boxes}")


if __name__ == '__main__':
    main()
