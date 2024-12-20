import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, random_split


def show_img(img, boxes, labels, filename, class_names=None):
    """
    Display an image with bounding boxes drawn around detected traffic signs,
    and display the image filename on top.

    Args:
        img (Tensor): The image tensor (C, H, W), typically from DataLoader.
        boxes (Tensor): Bounding boxes of shape (N, 4) where N is the number of boxes.
                        Each box is (center_x, center_y, width, height) in normalized coordinates.
        labels (Tensor): Class labels for the boxes, shape (N,)
        filename (str): The filename of the image, to be displayed on top of the image.
        class_names (list of str, optional): List of class names to map label ids to.
                                             If not provided, just label ids will be used.
    """
    # Convert the image tensor back to a NumPy array for display
    img = img.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

    # Plot the image
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)

    # Display the filename on top of the image
    ax.text(0.5, 1.02, filename, color='red', fontsize=14, weight='bold', ha='center', transform=ax.transAxes)

    # Draw the bounding boxes
    for i in range(boxes.shape[0]):
        # Convert from normalized coordinates to pixel coordinates
        h, w, _ = img.shape
        center_x, center_y, width, height = boxes[i]
        x = (center_x - width / 2) * w
        y = (center_y - height / 2) * h
        box_width = width * w
        box_height = height * h

        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Annotate with class label
        label = labels[i].item()
        if class_names:
            label_name = class_names[label]
        else:
            label_name = str(label)  # Just show label id if class names are not provided

        ax.text(x, y - 5, label_name, color='r', fontsize=12, weight='bold')

    # Display the plot
    plt.axis('off')
    plt.show()


def split_dataset(dataset, train_ratio=0.8, batch_size=4, shuffle=False, collate_fn=None):
    """
    Split the dataset into training and testing sets based on the given ratio.

    Args:
        dataset (Dataset): The full dataset to be split.
        train_ratio (float): The ratio of the dataset to be used for training. The rest will be used for testing.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset before splitting.
        collate_fn (function, optional): A custom collate function to merge samples into a batch.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the testing set.
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    # Split the dataset into train and test subsets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print("Random")
    # Create DataLoader for the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    # Create DataLoader for the testing set
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return train_loader, test_loader
